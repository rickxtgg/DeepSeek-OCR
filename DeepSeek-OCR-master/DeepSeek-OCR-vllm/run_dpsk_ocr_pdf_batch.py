"""
批量PDF文件OCR处理脚本
====================

功能：批量处理文件夹中的多个PDF文件，为每个PDF生成OCR结果

使用场景：
    - 批量处理多个PDF文档
    - 文档库的批量数字化
    - 大规模文档处理任务

输入：
    - INPUT_PATH: 包含多个PDF文件的文件夹路径
    
输出：
    - 每个PDF生成独立的输出文件夹
    - {pdf_name}.mmd: 最终Markdown结果
    - {pdf_name}_det.mmd: 带定位信息的结果
    - {pdf_name}_layouts.pdf: 带标注的PDF
    - images/: 提取的图片

特点：
    - 自动遍历文件夹中的所有PDF文件
    - 为每个PDF创建独立的输出目录
    - 支持断点续传（跳过已处理的文件）
    - 显示整体进度和统计信息

性能：
    - T4 GPU: 每个PDF页面约10-20秒
    - 支持并发处理多个页面
    - 自动内存管理

作者：DeepSeek AI
修改日期：2025-10-21
版本：v1.0
"""

import os
import glob
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

# CUDA 环境配置
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 导入配置和依赖
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

# 注册自定义模型
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


# ============================================================================
# 初始化 LLM 模型（全局单例）
# ============================================================================
print("🔧 正在初始化 LLM 模型...")
llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,  
    enforce_eager=False,
    trust_remote_code=True, 
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    disable_mm_preprocessor_cache=True,
)

# N-gram 防重复处理器配置
# ngram_size=20: 较小的值，适合PDF处理（防止表格重复）
# window_size=50: 检测窗口
# whitelist: <td>, </td> 允许在表格中重复
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=20, 
        window_size=50, 
        whitelist_token_ids={128821, 128822}
    )
]

# 采样参数配置
sampling_params = SamplingParams(
    temperature=0.0,  # 贪婪解码，确保确定性输出
    max_tokens=8192,  # 最大生成长度
    logits_processors=logits_processors,
    skip_special_tokens=False,  # 保留特殊标记
    include_stop_str_in_output=True,
)

print("✅ LLM 模型初始化完成！\n")


# ============================================================================
# 终端颜色输出类
# ============================================================================
class Colors:
    """终端颜色代码"""
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    RESET = '\033[0m'


# ============================================================================
# PDF 处理函数
# ============================================================================

def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    将PDF转换为高质量图片序列
    
    使用 PyMuPDF (fitz) 将PDF的每一页渲染为图片
    
    Args:
        pdf_path (str): PDF文件路径
        dpi (int): 分辨率，默认144
            - 72: 标准（快速预览）
            - 144: 推荐（平衡质量和速度）
            - 300: 高质量（适合打印）
        image_format (str): 输出格式，默认"PNG"
            - "PNG": 无损压缩
            - "JPEG": 有损压缩（文件更小）
    
    Returns:
        list: PIL Image 对象列表，每个元素对应PDF的一页
        
    内存管理:
        设置 Image.MAX_IMAGE_PIXELS = None 避免大图限制
        
    颜色空间处理:
        自动处理 RGBA → RGB 转换（使用白色背景）
    """
    images = []
    
    # 打开PDF文档
    pdf_document = fitz.open(pdf_path)
    
    # 计算缩放矩阵（DPI转换）
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    # 逐页渲染
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        
        # 渲染页面为像素图
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        
        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            # 处理透明通道
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
        
        images.append(img)
    
    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images, output_path):
    """
    将PIL图片列表转换为PDF文件
    
    用于生成带标注的PDF（_layouts.pdf）
    
    Args:
        pil_images (list): PIL Image 对象列表
        output_path (str): 输出PDF文件路径
        
    处理流程:
        1. 确保所有图片为RGB模式
        2. 转换为JPEG字节流（quality=95）
        3. 使用img2pdf合并为PDF
        
    质量设置:
        JPEG quality=95: 高质量但文件不会太大
    """
    if not pil_images:
        print(f"{Colors.YELLOW}警告: 没有图片可转换为PDF{Colors.RESET}")
        return
    
    image_bytes_list = []
    
    for img in pil_images:
        # 确保RGB模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 转换为JPEG字节流
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)
    
    try:
        # 合并为PDF
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"{Colors.GREEN}✓{Colors.RESET} 已生成带标注的PDF: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"{Colors.RED}✗ 生成PDF失败: {e}{Colors.RESET}")


def re_match(text):
    """
    从OCR结果中提取定位标记
    
    Args:
        text (str): OCR输出的原始文本
        
    Returns:
        tuple: (所有匹配项, 图片匹配项, 其他匹配项)
        
    标记格式:
        <|ref|>类型<|/ref|><|det|>坐标<|/det|>
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    mathes_image = []
    mathes_other = []
    
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """
    解析定位标记中的坐标信息
    
    Args:
        ref_text (tuple): 正则匹配的元组 (完整匹配, 类型, 坐标字符串)
        image_width (int): 图片宽度（像素）
        image_height (int): 图片高度（像素）
        
    Returns:
        tuple or None: (label_type, cor_list) 或 None（解析失败）
        
    坐标转换:
        归一化坐标（0-999）→ 实际像素坐标
    """
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(f"{Colors.YELLOW}警告: 坐标解析失败 - {e}{Colors.RESET}")
        return None
    return (label_type, cor_list)


def process_image_with_refs(image, refs, img_idx_offset=0, images_output_path=None):
    """
    在图片上绘制边界框并保存提取的图片
    
    Args:
        image (Image): PIL Image 对象
        refs (list): 定位标记列表
        img_idx_offset (int): 图片索引偏移量（用于多PDF批量处理）
        images_output_path (str): 图片保存路径，如果为None则不保存图片
        
    Returns:
        Image: 绘制了边界框的图片
        
    绘制效果:
        - 彩色边框（每种类型随机颜色）
        - 半透明填充
        - 类型标签
        - 特殊处理标题（粗边框）
    """
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # 创建半透明覆盖层
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    font = ImageFont.load_default()
    img_idx = img_idx_offset
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                # 生成随机颜色
                color = (
                    np.random.randint(0, 200), 
                    np.random.randint(0, 200), 
                    np.random.randint(0, 255)
                )
                color_a = color + (20,)  # 添加透明度
                
                for points in points_list:
                    x1, y1, x2, y2 = points
                    # 归一化坐标 → 像素坐标
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)
                    
                    # 如果是图片区域，裁剪并保存
                    if label_type == 'image' and images_output_path:
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{images_output_path}/{img_idx}.jpg")
                        except Exception as e:
                            print(f"{Colors.YELLOW}警告: 图片裁剪失败 - {e}{Colors.RESET}")
                        img_idx += 1
                    
                    # 绘制边框
                    width = 4 if label_type == 'title' else 2
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                    draw2.rectangle([x1, y1, x2, y2], fill=color_a)
                    
                    # 绘制标签
                    text_x = x1
                    text_y = max(0, y1 - 15)
                    draw.text((text_x, text_y), label_type, font=font, fill=color)
        except Exception as e:
            continue
    
    # 合并覆盖层
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_single_image(image):
    """
    预处理单张图片（多线程版本）
    
    Args:
        image (Image): PIL Image 对象
        
    Returns:
        dict: 包含提示词和图像特征的字典
    """
    prompt_in = PROMPT
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image], 
                bos=True, 
                eos=True, 
                cropping=CROP_MODE
            )
        },
    }
    return cache_item


def process_single_pdf(pdf_path, output_base_path):
    """
    处理单个PDF文件
    
    Args:
        pdf_path (str): PDF文件路径
        output_base_path (str): 输出基础路径
        
    Returns:
        dict: 处理结果统计信息
        
    处理流程:
        1. PDF → 图片序列
        2. 多线程预处理
        3. 批量OCR推理
        4. 提取定位信息
        5. 生成标注PDF
        6. 保存所有结果
        
    输出文件:
        - {pdf_name}.mmd: 最终Markdown
        - {pdf_name}_det.mmd: 带定位信息
        - {pdf_name}_layouts.pdf: 带标注的PDF
        - images/: 提取的图片
    """
    pdf_name = Path(pdf_path).stem
    start_time = time.time()
    
    print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.CYAN}📄 处理PDF: {pdf_name}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
    
    # 创建该PDF的输出目录
    pdf_output_path = os.path.join(output_base_path, pdf_name)
    os.makedirs(pdf_output_path, exist_ok=True)
    os.makedirs(f'{pdf_output_path}/images', exist_ok=True)
    
    # 检查是否已处理（断点续传）
    mmd_path = os.path.join(pdf_output_path, f'{pdf_name}.mmd')
    if os.path.exists(mmd_path):
        print(f"{Colors.YELLOW}⚠ 该PDF已处理，跳过: {pdf_name}{Colors.RESET}")
        return {
            'status': 'skipped',
            'pdf_name': pdf_name,
            'reason': 'already_processed'
        }
    
    try:
        # 1. PDF转图片
        print(f"{Colors.BLUE}📖 正在加载PDF...{Colors.RESET}")
        images = pdf_to_images_high_quality(pdf_path)
        print(f"{Colors.GREEN}✓{Colors.RESET} 已加载 {len(images)} 页")
        
        # 2. 多线程预处理
        print(f"{Colors.BLUE}🔄 正在预处理图片...{Colors.RESET}")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
            batch_inputs = list(tqdm(
                executor.map(process_single_image, images),
                total=len(images),
                desc=f"预处理 {pdf_name}",
                colour='blue'
            ))
        
        # 3. 批量OCR推理
        print(f"{Colors.BLUE}🤖 正在执行OCR识别...{Colors.RESET}")
        outputs_list = llm.generate(
            batch_inputs,
            sampling_params=sampling_params
        )
        
        # 4. 后处理结果
        print(f"{Colors.BLUE}📝 正在后处理结果...{Colors.RESET}")
        
        mmd_det_path = os.path.join(pdf_output_path, f'{pdf_name}_det.mmd')
        pdf_out_path = os.path.join(pdf_output_path, f'{pdf_name}_layouts.pdf')
        
        contents_det = ''  # 带定位标记的完整内容
        contents = ''      # 最终Markdown内容
        draw_images = []   # 带边界框的图片列表
        jdx = 0
        processed_pages = 0
        
        for output, img in zip(outputs_list, images):
            content = output.outputs[0].text
            
            # 检测重复页（如果没有正常结束符）
            if '<｜end▁of▁sentence｜>' in content:
                content = content.replace('<｜end▁of▁sentence｜>', '')
            else:
                if SKIP_REPEAT:
                    print(f"{Colors.YELLOW}⚠ 跳过重复页{Colors.RESET}")
                    continue
            
            processed_pages += 1
            
            # 添加页面分隔符
            page_num = f'\n<--- Page Split --->\n'
            contents_det += content + page_num
            
            # 提取定位信息并绘制边界框
            image_draw = img.copy()
            matches_ref, matches_images, mathes_other = re_match(content)
            # 传入当前PDF的图片保存路径
            images_save_path = f'{pdf_output_path}/images'
            result_image = process_image_with_refs(image_draw, matches_ref, jdx, images_save_path)
            draw_images.append(result_image)
            
            # 替换图片标记为Markdown图片链接
            for idx, a_match_image in enumerate(matches_images):
                content = content.replace(
                    a_match_image, 
                    f'![](images/{str(jdx)}_{str(idx)}.jpg)\n'
                )
            
            # 移除定位标记，清理格式
            for idx, a_match_other in enumerate(mathes_other):
                content = content.replace(a_match_other, '') \
                               .replace('\\coloneqq', ':=') \
                               .replace('\\eqqcolon', '=:') \
                               .replace('\n\n\n\n', '\n\n') \
                               .replace('\n\n\n', '\n\n')
            
            contents += content + page_num
            jdx += 1
        
        # 5. 保存所有结果
        with open(mmd_det_path, 'w', encoding='utf-8') as f:
            f.write(contents_det)
        
        with open(mmd_path, 'w', encoding='utf-8') as f:
            f.write(contents)
        
        pil_to_pdf_img2pdf(draw_images, pdf_out_path)
        
        # 计算处理时间
        elapsed_time = time.time() - start_time
        
        print(f"\n{Colors.GREEN}{'='*70}{Colors.RESET}")
        print(f"{Colors.GREEN}✅ PDF处理完成: {pdf_name}{Colors.RESET}")
        print(f"{Colors.GREEN}   总页数: {len(images)}{Colors.RESET}")
        print(f"{Colors.GREEN}   处理页数: {processed_pages}{Colors.RESET}")
        print(f"{Colors.GREEN}   耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟){Colors.RESET}")
        print(f"{Colors.GREEN}   平均速度: {elapsed_time/processed_pages:.2f} 秒/页{Colors.RESET}")
        print(f"{Colors.GREEN}   输出目录: {pdf_output_path}{Colors.RESET}")
        print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")
        
        return {
            'status': 'success',
            'pdf_name': pdf_name,
            'total_pages': len(images),
            'processed_pages': processed_pages,
            'elapsed_time': elapsed_time,
            'output_path': pdf_output_path
        }
        
    except Exception as e:
        print(f"\n{Colors.RED}{'='*70}{Colors.RESET}")
        print(f"{Colors.RED}❌ PDF处理失败: {pdf_name}{Colors.RESET}")
        print(f"{Colors.RED}   错误信息: {str(e)}{Colors.RESET}")
        print(f"{Colors.RED}{'='*70}{Colors.RESET}")
        
        return {
            'status': 'failed',
            'pdf_name': pdf_name,
            'error': str(e)
        }


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    """
    批量PDF文件OCR主流程
    
    功能:
        - 自动遍历文件夹中的所有PDF文件
        - 为每个PDF创建独立的输出目录
        - 支持断点续传（跳过已处理的文件）
        - 显示整体进度和统计信息
        
    使用方法:
        1. 在 config.py 中设置:
           INPUT_PATH = '/path/to/pdf/folder'  # PDF文件夹路径
           OUTPUT_PATH = '/path/to/output'     # 输出根目录
           
        2. 运行脚本:
           python run_dpsk_ocr_pdf_batch.py
           
        3. 查看结果:
           每个PDF的结果在: OUTPUT_PATH/{pdf_name}/
           
    输出结构:
        OUTPUT_PATH/
        ├── pdf1/
        │   ├── pdf1.mmd          # 最终Markdown结果
        │   ├── pdf1_det.mmd      # 带定位信息的结果
        │   ├── pdf1_layouts.pdf  # 带标注的PDF
        │   └── images/           # 从该PDF中提取的图片
        │       ├── 0.jpg
        │       ├── 1.jpg
        │       └── ...
        ├── pdf2/
        │   ├── pdf2.mmd
        │   ├── pdf2_det.mmd
        │   ├── pdf2_layouts.pdf
        │   └── images/
        └── ...
    """
    
    print(f"\n{Colors.MAGENTA}{'='*70}{Colors.RESET}")
    print(f"{Colors.MAGENTA}🚀 DeepSeek-OCR 批量PDF处理系统{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'='*70}{Colors.RESET}\n")
    
    # 创建输出根目录
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # 注意：不在根目录创建 images/ 目录，每个PDF的图片保存在各自的子目录中
    
    # 获取所有PDF文件
    print(f"{Colors.BLUE}📂 正在扫描PDF文件...{Colors.RESET}")
    pdf_files = glob.glob(os.path.join(INPUT_PATH, '*.pdf'))
    
    if not pdf_files:
        print(f"{Colors.RED}❌ 错误: 在 {INPUT_PATH} 中没有找到PDF文件{Colors.RESET}")
        exit(1)
    
    print(f"{Colors.GREEN}✓ 找到 {len(pdf_files)} 个PDF文件{Colors.RESET}")
    
    # 显示文件列表
    print(f"\n{Colors.CYAN}PDF文件列表:{Colors.RESET}")
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_name = Path(pdf_file).name
        print(f"  {i}. {pdf_name}")
    
    print(f"\n{Colors.BLUE}开始批量处理...{Colors.RESET}\n")
    
    # 统计信息
    total_start_time = time.time()
    results = []
    
    # 逐个处理PDF
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        print(f"{Colors.MAGENTA}📊 总进度: {i}/{len(pdf_files)}{Colors.RESET}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        
        result = process_single_pdf(pdf_file, OUTPUT_PATH)
        results.append(result)
    
    # 处理完成，显示总结
    total_elapsed_time = time.time() - total_start_time
    
    print(f"\n{Colors.MAGENTA}{'='*70}{Colors.RESET}")
    print(f"{Colors.MAGENTA}🎉 批量处理完成！{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'='*70}{Colors.RESET}\n")
    
    # 统计结果
    success_count = len([r for r in results if r['status'] == 'success'])
    failed_count = len([r for r in results if r['status'] == 'failed'])
    skipped_count = len([r for r in results if r['status'] == 'skipped'])
    
    total_pages = sum([r.get('total_pages', 0) for r in results if r['status'] == 'success'])
    processed_pages = sum([r.get('processed_pages', 0) for r in results if r['status'] == 'success'])
    
    print(f"{Colors.CYAN}📊 处理统计:{Colors.RESET}")
    print(f"  总PDF数: {len(pdf_files)}")
    print(f"  {Colors.GREEN}✓ 成功: {success_count}{Colors.RESET}")
    print(f"  {Colors.RED}✗ 失败: {failed_count}{Colors.RESET}")
    print(f"  {Colors.YELLOW}⊘ 跳过: {skipped_count}{Colors.RESET}")
    
    if success_count > 0:
        print(f"\n{Colors.CYAN}📄 页面统计:{Colors.RESET}")
        print(f"  总页数: {total_pages}")
        print(f"  处理页数: {processed_pages}")
        
        print(f"\n{Colors.CYAN}⏱️  时间统计:{Colors.RESET}")
        print(f"  总耗时: {total_elapsed_time:.2f} 秒 ({total_elapsed_time/60:.2f} 分钟)")
        print(f"  平均速度: {total_elapsed_time/processed_pages:.2f} 秒/页")
        print(f"  吞吐量: {processed_pages/(total_elapsed_time/60):.2f} 页/分钟")
    
    # 显示失败的文件
    if failed_count > 0:
        print(f"\n{Colors.RED}❌ 失败的PDF:{Colors.RESET}")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['pdf_name']}: {r.get('error', 'Unknown error')}")
    
    # 显示跳过的文件
    if skipped_count > 0:
        print(f"\n{Colors.YELLOW}⊘ 跳过的PDF (已处理):{Colors.RESET}")
        for r in results:
            if r['status'] == 'skipped':
                print(f"  - {r['pdf_name']}")
    
    print(f"\n{Colors.GREEN}✅ 所有结果已保存到: {OUTPUT_PATH}{Colors.RESET}")
    print(f"\n{Colors.MAGENTA}{'='*70}{Colors.RESET}\n")

