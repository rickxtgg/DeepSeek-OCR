#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-OCR T4 GPU 自动修复脚本
================================

功能：自动修复DeepSeek-OCR项目中的T4 GPU兼容性问题和vLLM版本兼容性问题

修复内容：
1. T4 GPU兼容性修复:
   - 添加 dtype='half' 参数（替换bfloat16为float16）
   - 修改 block_size 从256到16
   - 修复视觉编码器的dtype转换
   - 修复输入数据的dtype转换

2. vLLM版本兼容性修复:
   - SamplingMetadata 导入兼容
   - set_default_torch_dtype 导入兼容
   - merge_multimodal_embeddings 导入兼容
   - ModelRegistry 导入兼容
   - AsyncLLMEngine/AsyncEngineArgs 导入兼容

使用方法：
    python auto_fix_t4_compatibility.py [项目路径]
    
    如果不指定路径，默认处理当前目录下的 DeepSeek-OCR-master 文件夹

示例：
    python auto_fix_t4_compatibility.py
    python auto_fix_t4_compatibility.py /path/to/DeepSeek-OCR

作者：DeepSeek AI & Contributors
日期：2025-10-21
版本：v3.0
"""

import os
import sys
import shutil
import re
from pathlib import Path
from datetime import datetime


class Colors:
    """终端颜色代码"""
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class T4CompatibilityFixer:
    """
    T4 GPU 兼容性修复器
    
    功能描述：
        提供完整的T4 GPU兼容性修复功能和vLLM版本兼容性修复
        自动检测、备份、修复和验证所有必需的文件
    
    主要功能：
        - 环境检查和文件验证
        - 自动备份原始文件
        - 应用T4 GPU兼容性修复
        - 应用vLLM版本兼容性修复
        - 修复结果验证
        - 恢复备份功能
        - 生成详细报告
    
    修复内容：
        T4 GPU修复:
        1. dtype='half': 在引擎参数中添加float16支持
        2. block_size: 从256改为16
        3. 视觉编码器: 转换到float16
        4. 输入数据: 动态dtype转换
        
        vLLM版本兼容性修复:
        1. SamplingMetadata 导入兼容
        2. set_default_torch_dtype 导入兼容
        3. merge_multimodal_embeddings 导入兼容
        4. ModelRegistry 导入兼容
        5. AsyncLLMEngine/AsyncEngineArgs 导入兼容
    
    使用方法：
        fixer = T4CompatibilityFixer()
        fixer.run_interactive()
    """
    
    def __init__(self, project_path=None):
        """
        初始化修复器
        
        参数：
            project_path (str, 可选): 项目路径，默认为当前目录下的 DeepSeek-OCR-master
        """
        if project_path is None:
            project_path = os.path.join(os.getcwd(), 'DeepSeek-OCR-master')
        
        self.project_path = Path(project_path)
        self.vllm_path = self.project_path / 'DeepSeek-OCR-vllm'
        self.backup_dir = self.project_path / f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # 需要修复的文件列表（OCR脚本和核心模型）
        self.files_to_fix = [
            'run_dpsk_ocr_image.py',
            'run_dpsk_ocr_pdf.py',
            'run_dpsk_ocr_eval_batch.py',
            'run_dpsk_ocr_pdf_batch.py',
            'deepseek_ocr.py'
        ]
        
        # 共享模块列表（被多个脚本共同使用）
        self.shared_modules = [
            'deepseek_ocr.py',
            'process/image_process.py',
        ]
        
        # 配置文件列表
        self.config_files = [
            'config.py',
            'config_image.py',
            'config_pdf.py',
            'config_batch.py',
            'config_pdf_batch.py'
        ]
        
        # OCR脚本与配置文件的映射关系
        self.script_config_mapping = {
            'run_dpsk_ocr_image.py': 'config_image',
            'run_dpsk_ocr_pdf.py': 'config_pdf',
            'run_dpsk_ocr_eval_batch.py': 'config_batch',
            'run_dpsk_ocr_pdf_batch.py': 'config_pdf_batch',
        }
        
        # 新创建的配置文件（用于恢复时删除）
        self.created_config_files = [
            'config_image.py',
            'config_pdf.py',
            'config_batch.py',
            'config_pdf_batch.py'
        ]
        
        # 新创建的目录（用于恢复时可选删除）
        self.created_directories = [
            'input_image', 'output_image',
            'input_pdf', 'output_pdf',
            'input_batch', 'output_batch',
            'input_pdf_batch', 'output_pdf_batch',
            'input', 'output'
        ]
        
        # 修复统计
        self.stats = {
            'total_files': 0,
            'fixed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'fixes_applied': 0
        }
    
    def check_environment(self):
        """检查环境和文件"""
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔍 检查环境{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        # 检查项目路径
        if not self.project_path.exists():
            print(f"{Colors.RED}❌ 错误: 项目路径不存在: {self.project_path}{Colors.RESET}")
            return False
        print(f"{Colors.GREEN}✓{Colors.RESET} 项目路径: {self.project_path}")
        
        # 检查vllm路径
        if not self.vllm_path.exists():
            print(f"{Colors.RED}❌ 错误: vllm路径不存在: {self.vllm_path}{Colors.RESET}")
            return False
        print(f"{Colors.GREEN}✓{Colors.RESET} vLLM路径: {self.vllm_path}")
        
        # 检查需要修复的文件
        print(f"\n{Colors.BLUE}📄 待修复文件:{Colors.RESET}")
        existing_files = []
        for filename in self.files_to_fix:
            filepath = self.vllm_path / filename
            if filepath.exists():
                existing_files.append(filename)
                print(f"  {Colors.GREEN}✓{Colors.RESET} {filename}")
            else:
                print(f"  {Colors.YELLOW}⊘{Colors.RESET} {filename} (不存在，跳过)")
        
        if not existing_files:
            print(f"\n{Colors.RED}❌ 错误: 没有找到需要修复的文件{Colors.RESET}")
            return False
        
        self.stats['total_files'] = len(existing_files)
        return True
    
    def create_backup(self, include_configs=True):
        """创建备份（包括脚本文件、共享模块和配置文件）"""
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}💾 创建备份{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            print(f"{Colors.BLUE}备份目录: {self.backup_dir}{Colors.RESET}\n")
            
            backup_count = 0
            
            # 备份 OCR 脚本文件
            print(f"{Colors.CYAN}备份脚本文件:{Colors.RESET}")
            for filename in self.files_to_fix:
                src_file = self.vllm_path / filename
                if src_file.exists():
                    dst_file = self.backup_dir / filename
                    shutil.copy2(src_file, dst_file)
                    print(f"  {Colors.GREEN}✓{Colors.RESET} {filename}")
                    backup_count += 1
            
            # 备份共享模块
            print(f"\n{Colors.CYAN}备份共享模块:{Colors.RESET}")
            for module_path in self.shared_modules:
                src_file = self.vllm_path / module_path
                if src_file.exists():
                    # 确保备份目录存在（处理子目录如 process/）
                    dst_file = self.backup_dir / module_path
                    os.makedirs(dst_file.parent, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    print(f"  {Colors.GREEN}✓{Colors.RESET} {module_path}")
                    backup_count += 1
            
            # 备份配置文件
            if include_configs:
                print(f"\n{Colors.CYAN}备份配置文件:{Colors.RESET}")
                for filename in self.config_files:
                    src_file = self.vllm_path / filename
                    if src_file.exists():
                        dst_file = self.backup_dir / filename
                        shutil.copy2(src_file, dst_file)
                        print(f"  {Colors.GREEN}✓{Colors.RESET} {filename}")
                        backup_count += 1
                    else:
                        # 记录原本不存在的配置文件（恢复时需要删除）
                        marker_file = self.backup_dir / f'.{filename}.not_exists'
                        marker_file.touch()
            
            # 保存备份元信息
            meta_file = self.backup_dir / '.backup_meta.txt'
            with open(meta_file, 'w', encoding='utf-8') as f:
                f.write(f"backup_time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"backup_count={backup_count}\n")
                f.write(f"include_configs={include_configs}\n")
            
            print(f"\n{Colors.GREEN}✅ 备份完成！共备份 {backup_count} 个文件{Colors.RESET}")
            return True
        except Exception as e:
            print(f"\n{Colors.RED}❌ 备份失败: {e}{Colors.RESET}")
            return False
    
    def list_backups(self):
        """列出所有备份目录"""
        backups = []
        if self.project_path.exists():
            for item in self.project_path.iterdir():
                if item.is_dir() and item.name.startswith('backup_'):
                    backups.append(item)
        return sorted(backups, reverse=True)  # 最新的在前面
    
    def restore_from_backup(self, backup_dir=None):
        """从备份恢复文件（恢复所有修改，删除新创建的文件）"""
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔄 恢复备份{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        backups = self.list_backups()
        
        if not backups:
            print(f"{Colors.RED}❌ 错误: 没有找到任何备份目录{Colors.RESET}")
            return False
        
        if backup_dir is None:
            # 显示可用的备份
            print(f"{Colors.BLUE}可用的备份:{Colors.RESET}\n")
            for i, backup in enumerate(backups, 1):
                # 解析时间戳
                timestamp = backup.name.replace('backup_', '')
                try:
                    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_time = timestamp
                
                # 读取备份元信息
                meta_file = backup / '.backup_meta.txt'
                file_count = "未知"
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        for line in f:
                            if line.startswith('backup_count='):
                                file_count = line.split('=')[1].strip()
                
                print(f"  {i}. {backup.name} ({formatted_time}, {file_count}个文件)")
            
            print(f"\n请选择要恢复的备份编号 (1-{len(backups)})，留空取消: ", end='')
            choice = input().strip()
            
            if not choice:
                print(f"{Colors.YELLOW}⊘{Colors.RESET} 已取消恢复操作")
                return False
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(backups):
                    backup_dir = backups[idx]
                else:
                    print(f"{Colors.RED}❌ 无效的选择{Colors.RESET}")
                    return False
            except ValueError:
                print(f"{Colors.RED}❌ 无效的输入{Colors.RESET}")
                return False
        
        print(f"\n{Colors.BLUE}从备份恢复: {backup_dir}{Colors.RESET}\n")
        
        try:
            restored_count = 0
            deleted_count = 0
            
            # 1. 恢复脚本文件
            print(f"{Colors.CYAN}恢复脚本文件:{Colors.RESET}")
            for filename in self.files_to_fix:
                backup_file = backup_dir / filename
                target_file = self.vllm_path / filename
                
                if backup_file.exists():
                    shutil.copy2(backup_file, target_file)
                    print(f"  {Colors.GREEN}✓{Colors.RESET} 已恢复: {filename}")
                    restored_count += 1
            
            # 2. 恢复共享模块
            print(f"\n{Colors.CYAN}恢复共享模块:{Colors.RESET}")
            for module_path in self.shared_modules:
                backup_file = backup_dir / module_path
                target_file = self.vllm_path / module_path
                
                if backup_file.exists():
                    shutil.copy2(backup_file, target_file)
                    print(f"  {Colors.GREEN}✓{Colors.RESET} 已恢复: {module_path}")
                    restored_count += 1
                else:
                    # 检查是否有 .backup_shared 文件
                    shared_backup = str(target_file) + '.backup_shared'
                    if os.path.exists(shared_backup):
                        shutil.copy2(shared_backup, target_file)
                        print(f"  {Colors.GREEN}✓{Colors.RESET} 已从 .backup_shared 恢复: {module_path}")
                        restored_count += 1
            
            # 4. 恢复配置文件
            print(f"\n{Colors.CYAN}恢复配置文件:{Colors.RESET}")
            for filename in self.config_files:
                backup_file = backup_dir / filename
                target_file = self.vllm_path / filename
                marker_file = backup_dir / f'.{filename}.not_exists'
                
                if marker_file.exists():
                    # 原本不存在的文件，需要删除
                    if target_file.exists():
                        os.remove(target_file)
                        print(f"  {Colors.YELLOW}✗{Colors.RESET} 已删除（原本不存在）: {filename}")
                        deleted_count += 1
                elif backup_file.exists():
                    shutil.copy2(backup_file, target_file)
                    print(f"  {Colors.GREEN}✓{Colors.RESET} 已恢复: {filename}")
                    restored_count += 1
            
            # 5. 询问是否删除新创建的目录
            print(f"\n{Colors.YELLOW}是否删除新创建的输入输出目录？(y/n，默认n): {Colors.RESET}", end='')
            delete_dirs = input().strip().lower() == 'y'
            
            if delete_dirs:
                print(f"\n{Colors.CYAN}删除新创建的目录:{Colors.RESET}")
                for dirname in self.created_directories:
                    dir_path = self.vllm_path / dirname
                    if dir_path.exists() and dir_path.is_dir():
                        # 只删除空目录或询问确认
                        if not any(dir_path.iterdir()):
                            shutil.rmtree(dir_path)
                            print(f"  {Colors.YELLOW}✗{Colors.RESET} 已删除目录: {dirname}/")
                        else:
                            print(f"  {Colors.YELLOW}⊘{Colors.RESET} 目录非空，跳过: {dirname}/")
            
            print(f"\n{Colors.GREEN}{'='*70}{Colors.RESET}")
            print(f"{Colors.GREEN}✅ 恢复完成！{Colors.RESET}")
            print(f"  恢复文件数: {restored_count}")
            print(f"  删除文件数: {deleted_count}")
            print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")
            return True
                
        except Exception as e:
            print(f"\n{Colors.RED}❌ 恢复失败: {e}{Colors.RESET}")
            return False
    
    # ========================================================================
    # vLLM 版本兼容性修复方法
    # ========================================================================
    
    def fix_vllm_imports_deepseek_ocr(self, filepath):
        """修复 deepseek_ocr.py 中的 vLLM 导入兼容性"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复1: SamplingMetadata 导入
        old_import_1 = "from vllm.model_executor import SamplingMetadata"
        new_import_1 = """# 兼容旧版和新版 vllm 的 SamplingMetadata 导入
# 尝试多个可能的导入路径以确保兼容性
try:
    # 新版 vllm (0.6.0+): SamplingMetadata 在 sampling_metadata 子模块中
    from vllm.model_executor.sampling_metadata import SamplingMetadata
except ImportError:
    try:
        # 旧版 vllm: SamplingMetadata 直接从 model_executor 导入
        from vllm.model_executor import SamplingMetadata
    except ImportError:
        try:
            # 某些版本: 从 sequence 模块导入
            from vllm.sequence import SamplingMetadata
        except ImportError:
            try:
                # v1 API: 从 v1.sample.metadata 导入
                from vllm.v1.sample.metadata import SamplingMetadata
            except ImportError:
                # 如果所有导入都失败，抛出清晰的错误信息
                raise ImportError(
                    "无法导入 SamplingMetadata。请检查 vllm 版本，"
                    "尝试的导入路径：\\n"
                    "  - vllm.model_executor.sampling_metadata\\n"
                    "  - vllm.model_executor\\n"
                    "  - vllm.sequence\\n"
                    "  - vllm.v1.sample.metadata\\n"
                    "建议：pip install --upgrade vllm 或检查 vllm 版本兼容性"
                )"""
        
        if old_import_1 in content and "# 兼容旧版和新版 vllm 的 SamplingMetadata 导入" not in content:
            content = content.replace(old_import_1, new_import_1)
            fixes.append('SamplingMetadata 导入兼容')
        
        # 修复2: set_default_torch_dtype 导入
        old_import_2 = "from vllm.model_executor.model_loader.utils import set_default_torch_dtype"
        new_import_2 = """# 兼容旧版和新版 vllm 的 set_default_torch_dtype 导入
# 注意：此函数在代码中可能未使用，但保留导入以保持兼容性
try:
    # 新版 vllm: set_default_torch_dtype 在 utils.torch_utils 中
    from vllm.utils.torch_utils import set_default_torch_dtype
except ImportError:
    try:
        # 旧版 vllm: set_default_torch_dtype 在 model_loader.utils 中
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    except ImportError:
        # 如果都失败，尝试从其他可能的位置导入
        try:
            from vllm.utils import set_default_torch_dtype
        except ImportError:
            # 如果所有导入都失败，创建一个占位符或使用 torch 的默认行为
            # 由于代码中可能未使用此函数，我们创建一个 no-op 函数
            def set_default_torch_dtype(dtype):
                \"\"\"占位符函数，如果导入失败则使用此函数\"\"\"
                pass"""
        
        if old_import_2 in content and "# 兼容旧版和新版 vllm 的 set_default_torch_dtype 导入" not in content:
            content = content.replace(old_import_2, new_import_2)
            fixes.append('set_default_torch_dtype 导入兼容')
        
        # 修复3: merge_multimodal_embeddings 导入
        old_import_3 = """from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)"""
        new_import_3 = """# 兼容旧版和新版 vllm 的导入
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix)
# 兼容旧版和新版 vllm 的 merge_multimodal_embeddings 导入
try:
    # 旧版 vllm: merge_multimodal_embeddings 是公开函数
    from vllm.model_executor.models.utils import merge_multimodal_embeddings
except ImportError:
    try:
        # 新版 vllm: 可能是私有函数 _merge_multimodal_embeddings
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings as merge_multimodal_embeddings
    except ImportError:
        try:
            # 某些版本: 可能在其他位置
            from vllm.multimodal.utils import merge_multimodal_embeddings
        except ImportError:
            # 如果所有导入都失败，抛出清晰的错误信息
            raise ImportError(
                "无法导入 merge_multimodal_embeddings。请检查 vllm 版本，"
                "尝试的导入路径：\\n"
                "  - vllm.model_executor.models.utils.merge_multimodal_embeddings\\n"
                "  - vllm.model_executor.models.utils._merge_multimodal_embeddings\\n"
                "  - vllm.multimodal.utils.merge_multimodal_embeddings\\n"
                "建议：pip install --upgrade vllm 或检查 vllm 版本兼容性"
            )"""
        
        if old_import_3 in content and "# 兼容旧版和新版 vllm 的 merge_multimodal_embeddings 导入" not in content:
            content = content.replace(old_import_3, new_import_3)
            fixes.append('merge_multimodal_embeddings 导入兼容')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def fix_vllm_imports_run_scripts(self, filepath):
        """修复运行脚本中的 vLLM 导入兼容性 (ModelRegistry)"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复 ModelRegistry 导入
        old_import = "from vllm.model_executor.models.registry import ModelRegistry"
        new_import = """# 兼容旧版和新版 vllm 的 ModelRegistry 导入
try:
    from vllm.model_executor.models.registry import ModelRegistry
except ImportError:
    try:
        from vllm.model_executor.models import ModelRegistry
    except ImportError:
        from vllm.model_executor.model_loader import ModelRegistry"""
        
        if old_import in content and "# 兼容旧版和新版 vllm 的 ModelRegistry 导入" not in content:
            content = content.replace(old_import, new_import)
            fixes.append('ModelRegistry 导入兼容')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def fix_vllm_imports_run_image(self, filepath):
        """修复 run_dpsk_ocr_image.py 中的 vLLM 导入兼容性"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复 AsyncLLMEngine, AsyncEngineArgs, ModelRegistry 导入
        old_imports = """from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry"""
        
        new_imports = """# 兼容旧版和新版 vllm 的导入
try:
    from vllm import AsyncLLMEngine, SamplingParams
except ImportError:
    # 某些版本的 AsyncLLMEngine 可能在不同位置
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm import SamplingParams

try:
    from vllm.engine.arg_utils import AsyncEngineArgs
except ImportError:
    try:
        from vllm.engine.async_llm_engine import AsyncEngineArgs
    except ImportError:
        from vllm import AsyncEngineArgs

# 兼容旧版和新版 vllm 的 ModelRegistry 导入
try:
    from vllm.model_executor.models.registry import ModelRegistry
except ImportError:
    try:
        from vllm.model_executor.models import ModelRegistry
    except ImportError:
        from vllm.model_executor.model_loader import ModelRegistry"""
        
        if old_imports in content and "# 兼容旧版和新版 vllm 的导入" not in content:
            content = content.replace(old_imports, new_imports)
            fixes.append('AsyncLLMEngine/AsyncEngineArgs/ModelRegistry 导入兼容')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def fix_vllm_imports(self, filename):
        """修复单个文件的 vLLM 导入兼容性"""
        filepath = self.vllm_path / filename
        
        if not filepath.exists():
            return None
        
        print(f"\n{Colors.BLUE}📝 修复 vLLM 导入: {filename}{Colors.RESET}")
        
        try:
            if filename == 'deepseek_ocr.py':
                fixes = self.fix_vllm_imports_deepseek_ocr(filepath)
            elif filename == 'run_dpsk_ocr_image.py':
                fixes = self.fix_vllm_imports_run_image(filepath)
            elif filename in ['run_dpsk_ocr_pdf.py', 'run_dpsk_ocr_eval_batch.py', 'run_dpsk_ocr_pdf_batch.py']:
                fixes = self.fix_vllm_imports_run_scripts(filepath)
            else:
                fixes = None
            
            if fixes:
                self.stats['fixed_files'] += 1
                self.stats['fixes_applied'] += len(fixes)
                print(f"{Colors.GREEN}✓{Colors.RESET} 已修复 ({len(fixes)} 处):")
                for fix in fixes:
                    print(f"  • {fix}")
                return True
            else:
                print(f"{Colors.YELLOW}⊘{Colors.RESET} 已是最新或无需修复，跳过")
                return False
        
        except Exception as e:
            self.stats['failed_files'] += 1
            print(f"{Colors.RED}✗{Colors.RESET} 修复失败: {e}")
            return False
    
    # ========================================================================
    # T4 GPU 修复方法
    # ========================================================================
    
    def fix_run_dpsk_ocr_image(self, filepath):
        """修复 run_dpsk_ocr_image.py 的 T4 兼容性"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复1: block_size
        if 'block_size=256,' in content and 'block_size=16,' not in content:
            content = content.replace(
                'block_size=256,',
                'block_size=16,  # T4 GPU 修复: 256 不支持，改为 16'
            )
            fixes.append('block_size: 256 → 16')
        
        # 修复2: dtype='half' (AsyncEngineArgs)
        pattern = r"(gpu_memory_utilization=0\.75,\s*)\n(\s*)\)"
        if re.search(pattern, content) and "dtype='half'" not in content:
            replacement = r"\1\n\2dtype='half',  # 使用float16以支持compute capability 7.5的GPU (如Tesla T4)\n\2)"
            content = re.sub(pattern, replacement, content)
            fixes.append("dtype='half' (AsyncEngineArgs)")
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def fix_run_dpsk_ocr_pdf(self, filepath):
        """修复 run_dpsk_ocr_pdf.py 的 T4 兼容性"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复1: block_size
        if 'block_size=256,' in content and 'block_size=16,' not in content:
            content = content.replace(
                'block_size=256,',
                'block_size=16,  # T4 GPU 修复: 256 不支持，改为 16'
            )
            fixes.append('block_size: 256 → 16')
        
        # 修复2: 先确保 disable_mm_preprocessor_cache=True 后面有逗号
        if 'disable_mm_preprocessor_cache=True\n' in content:
            content = content.replace(
                'disable_mm_preprocessor_cache=True\n',
                'disable_mm_preprocessor_cache=True,\n'
            )
            fixes.append('添加逗号')
        
        # 修复3: dtype='half' (LLM)
        pattern = r"(disable_mm_preprocessor_cache=True,)\n(\s*)\)"
        if re.search(pattern, content) and "dtype='half'" not in content:
            replacement = r"\1\n\2dtype='half',  # 使用float16以支持compute capability 7.5的GPU (如Tesla T4)\n\2)"
            content = re.sub(pattern, replacement, content)
            fixes.append("dtype='half' (LLM)")
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def fix_run_dpsk_ocr_eval_batch(self, filepath):
        """修复 run_dpsk_ocr_eval_batch.py 的 T4 兼容性"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复1: block_size
        if 'block_size=256,' in content and 'block_size=16,' not in content:
            content = content.replace(
                'block_size=256,',
                'block_size=16,  # T4 GPU 修复: 256 不支持，改为 16'
            )
            fixes.append('block_size: 256 → 16')
        
        # 修复2: dtype='half' (LLM)
        pattern = r"(gpu_memory_utilization=0\.9,?\s*)\n(\s*)\)"
        if re.search(pattern, content) and "dtype='half'" not in content:
            replacement = r"\1\n\2dtype='half',  # 使用float16以支持compute capability 7.5的GPU (如Tesla T4)\n\2)"
            content = re.sub(pattern, replacement, content)
            fixes.append("dtype='half' (LLM)")
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def fix_run_dpsk_ocr_pdf_batch(self, filepath):
        """修复 run_dpsk_ocr_pdf_batch.py 的 T4 兼容性"""
        return self.fix_run_dpsk_ocr_pdf(filepath)
    
    def fix_deepseek_ocr(self, filepath):
        """修复 deepseek_ocr.py 的 T4 兼容性"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复1: 视觉编码器 dtype 转换
        old_code = """        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()

        n_embed = 1280
        self.projector =  MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos
    
        # self.sam_model = torch.compile(self.sam_model, mode="reduce-overhead")
        # self.vision_model = torch.compile(self.vision_model, mode="reduce-overhead")
        # self.projector = torch.compile(self.projector, mode="max-autotune")"""
        
        new_code = """        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()

        n_embed = 1280
        self.projector =  MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos
    
        # 修复 T4 GPU 兼容性：确保视觉编码器使用与主模型相同的 dtype
        # 当模型使用 float16 时，视觉编码器也需要转换为 float16
        target_dtype = model_config.dtype
        if target_dtype == torch.float16:
            self.sam_model = self.sam_model.to(dtype=torch.float16)
            self.vision_model = self.vision_model.to(dtype=torch.float16)
            self.projector = self.projector.to(dtype=torch.float16)
    
        # self.sam_model = torch.compile(self.sam_model, mode="reduce-overhead")
        # self.vision_model = torch.compile(self.vision_model, mode="reduce-overhead")
        # self.projector = torch.compile(self.projector, mode="max-autotune")"""
        
        if old_code in content and 'target_dtype' not in content:
            content = content.replace(old_code, new_code)
            fixes.append('视觉编码器 dtype 转换')
        
        # 修复2: 输入数据 dtype 转换
        old_pattern = r"(\s+)patches = images_crop\[jdx\]\[0\]\.to\(torch\.bfloat16\)\s*# batch_size = 1\n(\s+)image_ori = pixel_values\[jdx\]"
        
        if re.search(old_pattern, content) and 'model_dtype = next(self.sam_model.parameters()).dtype' not in content:
            new_pattern = r"\1# T4 GPU fix: 使用模型的实际 dtype 而不是硬编码 bfloat16\n\1model_dtype = next(self.sam_model.parameters()).dtype\n\1patches = images_crop[jdx][0].to(model_dtype) # batch_size = 1\n\2image_ori = pixel_values[jdx].to(model_dtype)"
            content = re.sub(old_pattern, new_pattern, content)
            fixes.append('输入数据 dtype 动态转换')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def fix_t4_file(self, filename):
        """修复单个文件的 T4 兼容性"""
        filepath = self.vllm_path / filename
        
        if not filepath.exists():
            self.stats['skipped_files'] += 1
            return None
        
        print(f"\n{Colors.BLUE}📝 修复 T4 兼容性: {filename}{Colors.RESET}")
        
        try:
            if filename == 'run_dpsk_ocr_image.py':
                fixes = self.fix_run_dpsk_ocr_image(filepath)
            elif filename == 'run_dpsk_ocr_pdf.py':
                fixes = self.fix_run_dpsk_ocr_pdf(filepath)
            elif filename == 'run_dpsk_ocr_eval_batch.py':
                fixes = self.fix_run_dpsk_ocr_eval_batch(filepath)
            elif filename == 'run_dpsk_ocr_pdf_batch.py':
                fixes = self.fix_run_dpsk_ocr_pdf_batch(filepath)
            elif filename == 'deepseek_ocr.py':
                fixes = self.fix_deepseek_ocr(filepath)
            else:
                fixes = None
            
            if fixes:
                self.stats['fixed_files'] += 1
                self.stats['fixes_applied'] += len(fixes)
                print(f"{Colors.GREEN}✓{Colors.RESET} 已修复 ({len(fixes)} 处):")
                for fix in fixes:
                    print(f"  • {fix}")
                return True
            else:
                self.stats['skipped_files'] += 1
                print(f"{Colors.YELLOW}⊘{Colors.RESET} 已是最新，跳过")
                return False
        
        except Exception as e:
            self.stats['failed_files'] += 1
            print(f"{Colors.RED}✗{Colors.RESET} 修复失败: {e}")
            return False
    
    def verify_fixes(self, verify_all=True, categories=None):
        """
        验证修复状态
        
        参数:
            verify_all: 是否验证所有功能
            categories: 要验证的功能列表，可选值：
                - 't4': T4 GPU 兼容性
                - 'vllm': vLLM 版本兼容性
                - 'config': 配置文件引用
                - 'memory': 内存优化
        """
        if categories is None:
            categories = ['t4', 'vllm', 'config', 'memory']
        
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔍 验证修复状态{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        
        all_results = {
            't4_fixes': [],
            'vllm_fixes': [],
            'config_refs': [],
            'config_files': [],
            'memory_fixes': []
        }
        
        section_num = 1
        
        # ========================================
        # 1. 验证 T4 GPU 兼容性修复
        # ========================================
        if 't4' in categories:
            print(f"\n{Colors.BLUE}【{section_num}】T4 GPU 兼容性修复状态{Colors.RESET}")
            print("-" * 50)
            section_num += 1
            
            for filename in self.files_to_fix:
                filepath = self.vllm_path / filename
                if not filepath.exists():
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if filename == 'deepseek_ocr.py':
                    checks = {
                        'target_dtype（视觉编码器）': 'target_dtype' in content,
                        'model_dtype（输入数据）': 'model_dtype = next(self.sam_model' in content
                    }
                else:
                    checks = {
                        'block_size=16': 'block_size=16' in content,
                        "dtype='half'": "dtype='half'" in content
                    }
                
                all_passed = all(checks.values())
                is_original = not any(checks.values())
                
                if is_original:
                    status = f"{Colors.YELLOW}○{Colors.RESET}"
                    status_text = "原始文件"
                elif all_passed:
                    status = f"{Colors.GREEN}✓{Colors.RESET}"
                    status_text = "已修复"
                else:
                    status = f"{Colors.RED}✗{Colors.RESET}"
                    status_text = "部分修复"
                
                print(f"  {status} {filename} ({status_text})")
                
                if not all_passed and not is_original:
                    for check, result in checks.items():
                        if not result:
                            print(f"      {Colors.RED}✗{Colors.RESET} {check}")
                
                all_results['t4_fixes'].append({
                    'filename': filename,
                    'passed': all_passed,
                    'is_original': is_original
                })
        
        # ========================================
        # 2. 验证 vLLM 版本兼容性修复
        # ========================================
        if 'vllm' in categories:
            print(f"\n{Colors.BLUE}【{section_num}】vLLM 版本兼容性修复状态{Colors.RESET}")
            print("-" * 50)
            section_num += 1
            
            vllm_checks = {
                'deepseek_ocr.py': [
                    ('SamplingMetadata兼容导入', '# 兼容旧版和新版 vllm 的 SamplingMetadata 导入'),
                    ('set_default_torch_dtype兼容导入', '# 兼容旧版和新版 vllm 的 set_default_torch_dtype 导入'),
                    ('merge_multimodal_embeddings兼容导入', '# 兼容旧版和新版 vllm 的 merge_multimodal_embeddings 导入')
                ],
                'run_dpsk_ocr_image.py': [
                    ('AsyncLLMEngine/ModelRegistry兼容导入', '# 兼容旧版和新版 vllm 的导入')
                ],
                'run_dpsk_ocr_pdf.py': [
                    ('ModelRegistry兼容导入', '# 兼容旧版和新版 vllm 的 ModelRegistry 导入')
                ],
                'run_dpsk_ocr_eval_batch.py': [
                    ('ModelRegistry兼容导入', '# 兼容旧版和新版 vllm 的 ModelRegistry 导入')
                ],
                'run_dpsk_ocr_pdf_batch.py': [
                    ('ModelRegistry兼容导入', '# 兼容旧版和新版 vllm 的 ModelRegistry 导入')
                ]
            }
            
            for filename, checks in vllm_checks.items():
                filepath = self.vllm_path / filename
                if not filepath.exists():
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_results = []
                for check_name, check_pattern in checks:
                    passed = check_pattern in content
                    file_results.append((check_name, passed))
                
                all_passed = all(r[1] for r in file_results)
                is_original = not any(r[1] for r in file_results)
                
                if is_original:
                    status = f"{Colors.YELLOW}○{Colors.RESET}"
                    status_text = "原始文件"
                elif all_passed:
                    status = f"{Colors.GREEN}✓{Colors.RESET}"
                    status_text = "已修复"
                else:
                    status = f"{Colors.RED}✗{Colors.RESET}"
                    status_text = "部分修复"
                
                print(f"  {status} {filename} ({status_text})")
                
                if not all_passed and not is_original:
                    for check_name, passed in file_results:
                        if not passed:
                            print(f"      {Colors.RED}✗{Colors.RESET} {check_name}")
                
                all_results['vllm_fixes'].append({
                    'filename': filename,
                    'passed': all_passed,
                    'is_original': is_original
                })
        
        # ========================================
        # 3. 验证配置文件引用
        # ========================================
        if 'config' in categories:
            print(f"\n{Colors.BLUE}【{section_num}】配置文件引用状态{Colors.RESET}")
            print("-" * 50)
            section_num += 1
            
            for script, expected_config in self.script_config_mapping.items():
                filepath = self.vllm_path / script
                if not filepath.exists():
                    print(f"  {Colors.YELLOW}⊘{Colors.RESET} {script} (文件不存在)")
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查当前使用的配置
                uses_expected = f'from {expected_config} import' in content
                uses_original = 'from config import' in content and f'from {expected_config}' not in content
                
                if uses_expected:
                    status = f"{Colors.GREEN}✓{Colors.RESET}"
                    config_used = expected_config
                    status_text = "独立配置"
                elif uses_original:
                    status = f"{Colors.YELLOW}○{Colors.RESET}"
                    config_used = "config"
                    status_text = "原始配置"
                else:
                    status = f"{Colors.RED}?{Colors.RESET}"
                    config_used = "未知"
                    status_text = "未知"
                
                print(f"  {status} {script}")
                print(f"      当前配置: {config_used}.py ({status_text})")
                print(f"      推荐配置: {expected_config}.py")
                
                all_results['config_refs'].append({
                    'script': script,
                    'expected_config': expected_config,
                    'uses_expected': uses_expected,
                    'uses_original': uses_original
                })
            
            # 验证配置文件存在性
            print(f"\n  {Colors.CYAN}配置文件存在性:{Colors.RESET}")
            
            for config_file in self.config_files:
                filepath = self.vllm_path / config_file
                exists = filepath.exists()
                
                if exists:
                    status = f"{Colors.GREEN}✓{Colors.RESET}"
                else:
                    status = f"{Colors.RED}✗{Colors.RESET}"
                
                print(f"    {status} {config_file} {'(存在)' if exists else '(不存在)'}")
                all_results['config_files'].append({
                    'filename': config_file,
                    'exists': exists
                })
        
        # ========================================
        # 4. 验证内存优化
        # ========================================
        if 'memory' in categories:
            print(f"\n{Colors.BLUE}【{section_num}】内存优化状态{Colors.RESET}")
            print("-" * 50)
            section_num += 1
            
            memory_checks = {
                'run_dpsk_ocr_pdf_batch.py': [
                    ('cleanup_memory() 函数', 'def cleanup_memory():'),
                    ('全局处理器单例', 'def get_processor():'),
                    ('分批处理 PAGE_BATCH_SIZE', 'PAGE_BATCH_SIZE'),
                    ('线程数限制', 'min(NUM_WORKERS'),
                    ('PDF间内存清理', '# 每处理完一个PDF就强制清理内存'),
                ],
                'run_dpsk_ocr_eval_batch.py': [
                    ('cleanup_memory() 函数', 'def cleanup_memory():'),
                    ('全局处理器单例', 'get_processor()'),
                    ('分批处理 BATCH_SIZE', 'BATCH_SIZE'),
                ],
                'run_dpsk_ocr_pdf.py': [
                    ('cleanup_memory() 函数', 'def cleanup_memory():'),
                    ('全局处理器单例', 'get_processor()'),
                ],
                'run_dpsk_ocr_image.py': [
                    ('cleanup_memory() 函数', 'def cleanup_memory():'),
                ],
            }
            
            for filename, checks in memory_checks.items():
                filepath = self.vllm_path / filename
                if not filepath.exists():
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_results = []
                for check_name, check_pattern in checks:
                    passed = check_pattern in content
                    file_results.append((check_name, passed))
                
                passed_count = sum(1 for r in file_results if r[1])
                total_count = len(file_results)
                is_original = passed_count == 0
                all_passed = passed_count == total_count
                
                if is_original:
                    status = f"{Colors.YELLOW}○{Colors.RESET}"
                    status_text = "未优化"
                elif all_passed:
                    status = f"{Colors.GREEN}✓{Colors.RESET}"
                    status_text = "已优化"
                else:
                    status = f"{Colors.YELLOW}△{Colors.RESET}"
                    status_text = f"部分优化 ({passed_count}/{total_count})"
                
                print(f"  {status} {filename} ({status_text})")
                
                if not all_passed and not is_original:
                    for check_name, passed in file_results:
                        if not passed:
                            print(f"      {Colors.RED}✗{Colors.RESET} {check_name}")
                
                all_results['memory_fixes'].append({
                    'filename': filename,
                    'passed': all_passed,
                    'is_original': is_original,
                    'passed_count': passed_count,
                    'total_count': total_count
                })
        
        # ========================================
        # 总结
        # ========================================
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        print(f"{Colors.MAGENTA}📊 验证总结{Colors.RESET}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        
        # T4 修复状态
        if 't4' in categories and all_results['t4_fixes']:
            t4_fixed = sum(1 for r in all_results['t4_fixes'] if r['passed'])
            t4_original = sum(1 for r in all_results['t4_fixes'] if r['is_original'])
            t4_total = len(all_results['t4_fixes'])
            print(f"\n  T4 GPU 兼容性: {t4_fixed}/{t4_total} 已修复, {t4_original} 原始文件")
        
        # vLLM 修复状态
        if 'vllm' in categories and all_results['vllm_fixes']:
            vllm_fixed = sum(1 for r in all_results['vllm_fixes'] if r['passed'])
            vllm_original = sum(1 for r in all_results['vllm_fixes'] if r['is_original'])
            vllm_total = len(all_results['vllm_fixes'])
            print(f"  vLLM 兼容性:   {vllm_fixed}/{vllm_total} 已修复, {vllm_original} 原始文件")
        
        # 配置引用状态
        if 'config' in categories and all_results['config_refs']:
            config_correct = sum(1 for r in all_results['config_refs'] if r['uses_expected'])
            config_original = sum(1 for r in all_results['config_refs'] if r['uses_original'])
            config_total = len(all_results['config_refs'])
            print(f"  配置文件引用: {config_correct}/{config_total} 使用独立配置, {config_original} 使用原始配置")
            
            # 配置文件存在性
            config_exists = sum(1 for r in all_results['config_files'] if r['exists'])
            config_files_total = len(all_results['config_files'])
            print(f"  配置文件存在: {config_exists}/{config_files_total}")
        
        # 内存优化状态
        if 'memory' in categories and all_results['memory_fixes']:
            mem_fixed = sum(1 for r in all_results['memory_fixes'] if r['passed'])
            mem_original = sum(1 for r in all_results['memory_fixes'] if r['is_original'])
            mem_total = len(all_results['memory_fixes'])
            print(f"  内存优化:      {mem_fixed}/{mem_total} 已优化, {mem_original} 未优化")
        
        # 整体状态判断
        all_original = True
        all_fixed = True
        
        if 't4' in categories and all_results['t4_fixes']:
            t4_fixed = sum(1 for r in all_results['t4_fixes'] if r['passed'])
            t4_original = sum(1 for r in all_results['t4_fixes'] if r['is_original'])
            t4_total = len(all_results['t4_fixes'])
            if t4_original != t4_total:
                all_original = False
            if t4_fixed != t4_total:
                all_fixed = False
        
        if 'vllm' in categories and all_results['vllm_fixes']:
            vllm_fixed = sum(1 for r in all_results['vllm_fixes'] if r['passed'])
            vllm_original = sum(1 for r in all_results['vllm_fixes'] if r['is_original'])
            vllm_total = len(all_results['vllm_fixes'])
            if vllm_original != vllm_total:
                all_original = False
            if vllm_fixed != vllm_total:
                all_fixed = False
        
        if 'memory' in categories and all_results['memory_fixes']:
            mem_fixed = sum(1 for r in all_results['memory_fixes'] if r['passed'])
            mem_original = sum(1 for r in all_results['memory_fixes'] if r['is_original'])
            mem_total = len(all_results['memory_fixes'])
            if mem_original != mem_total:
                all_original = False
            if mem_fixed != mem_total:
                all_fixed = False
        
        if all_original:
            print(f"\n{Colors.YELLOW}⚠️  所有文件都是原始状态，尚未应用任何修复{Colors.RESET}")
            return False
        elif all_fixed:
            print(f"\n{Colors.GREEN}✅ 所有修复已完成！{Colors.RESET}")
            return True
        else:
            print(f"\n{Colors.YELLOW}⚠️  部分修复已完成，建议运行完整修复{Colors.RESET}")
            return False
    
    def generate_report(self):
        """生成修复报告"""
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        print(f"{Colors.MAGENTA}📊 修复报告{Colors.RESET}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.RESET}\n")
        
        print(f"{Colors.CYAN}统计信息:{Colors.RESET}")
        print(f"  总文件数: {self.stats['total_files']}")
        print(f"  {Colors.GREEN}✓ 已修复: {self.stats['fixed_files']}{Colors.RESET}")
        print(f"  {Colors.YELLOW}⊘ 已跳过: {self.stats['skipped_files']}{Colors.RESET}")
        print(f"  {Colors.RED}✗ 修复失败: {self.stats['failed_files']}{Colors.RESET}")
        print(f"  修复总数: {self.stats['fixes_applied']} 处")
        
        print(f"\n{Colors.CYAN}备份位置:{Colors.RESET}")
        print(f"  {self.backup_dir}")
        
        print(f"\n{Colors.CYAN}下一步:{Colors.RESET}")
        print(f"  1. 验证修复是否生效")
        print(f"  2. 在T4 GPU上测试运行")
        print(f"  3. 如有问题，使用恢复功能恢复备份")
        
        # 保存报告到文件
        report_file = self.project_path / f'fix_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("DeepSeek-OCR T4 GPU 兼容性修复报告\n")
            f.write("="*70 + "\n\n")
            f.write(f"修复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"项目路径: {self.project_path}\n")
            f.write(f"备份路径: {self.backup_dir}\n\n")
            f.write("统计信息:\n")
            f.write(f"  总文件数: {self.stats['total_files']}\n")
            f.write(f"  已修复: {self.stats['fixed_files']}\n")
            f.write(f"  已跳过: {self.stats['skipped_files']}\n")
            f.write(f"  修复失败: {self.stats['failed_files']}\n")
            f.write(f"  修复总数: {self.stats['fixes_applied']} 处\n")
        
        print(f"\n{Colors.GREEN}📝 报告已保存: {report_file}{Colors.RESET}")
    
    def create_directories_and_update_config(self):
        """创建输入输出目录并更新config.py"""
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}📂 创建输入输出目录{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        try:
            input_dir = self.vllm_path / 'input'
            output_dir = self.vllm_path / 'output'
            
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"{Colors.GREEN}✓{Colors.RESET} 输入目录: {input_dir}")
            print(f"{Colors.GREEN}✓{Colors.RESET} 输出目录: {output_dir}")
            
            config_path = self.vllm_path / 'config.py'
            if config_path.exists():
                print(f"\n{Colors.BLUE}更新 config.py 路径配置...{Colors.RESET}")
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                input_path_str = str(input_dir).replace('\\', '/')
                output_path_str = str(output_dir).replace('\\', '/')
                
                if "INPUT_PATH = ''" in content or 'INPUT_PATH = ""' in content:
                    content = re.sub(
                        r"INPUT_PATH = ['\"].*?['\"]",
                        f"INPUT_PATH = '{input_path_str}'",
                        content
                    )
                    print(f"{Colors.GREEN}✓{Colors.RESET} 已更新 INPUT_PATH")
                
                if "OUTPUT_PATH = ''" in content or 'OUTPUT_PATH = ""' in content:
                    content = re.sub(
                        r"OUTPUT_PATH = ['\"].*?['\"]",
                        f"OUTPUT_PATH = '{output_path_str}'",
                        content
                    )
                    print(f"{Colors.GREEN}✓{Colors.RESET} 已更新 OUTPUT_PATH")
                
                if content != original_content:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                else:
                    print(f"{Colors.YELLOW}⊘{Colors.RESET} 路径已配置，无需更新")
            
            return True
        except Exception as e:
            print(f"{Colors.RED}✗{Colors.RESET} 创建目录失败: {e}")
            return False
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_files': 0,
            'fixed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'fixes_applied': 0
        }
    
    def show_menu(self):
        """显示交互式菜单"""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}🔧 DeepSeek-OCR 自动修复工具 v3.0{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}请选择要执行的操作:{Colors.RESET}\n")
        print(f"  {Colors.GREEN}1{Colors.RESET}. 完整修复 (T4 GPU + vLLM 兼容性)")
        print(f"  {Colors.GREEN}2{Colors.RESET}. 仅修复 T4 GPU 兼容性问题")
        print(f"  {Colors.GREEN}3{Colors.RESET}. 仅修复 vLLM 版本兼容性问题")
        print(f"  {Colors.GREEN}4{Colors.RESET}. 恢复备份 (撤销所有修改)")
        print(f"  {Colors.GREEN}5{Colors.RESET}. 验证当前修复状态")
        print(f"  {Colors.GREEN}6{Colors.RESET}. 创建独立配置文件 (图片/PDF/批量)")
        print(f"  {Colors.GREEN}7{Colors.RESET}. 添加内存优化 (防止OOM崩溃)")
        print(f"  {Colors.GREEN}0{Colors.RESET}. 退出")
        
        print(f"\n{Colors.YELLOW}提示: 直接按回车将不执行任何修改{Colors.RESET}")
        print(f"\n请输入选项 (0-7): ", end='')
        
        return input().strip()
    
    def show_verify_menu(self):
        """显示验证子菜单"""
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔍 验证修复状态{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}请选择要验证的功能:{Colors.RESET}\n")
        print(f"  {Colors.GREEN}1{Colors.RESET}. 验证全部 (T4 + vLLM + 配置 + 内存)")
        print(f"  {Colors.GREEN}2{Colors.RESET}. 仅验证 T4 GPU 兼容性修复")
        print(f"  {Colors.GREEN}3{Colors.RESET}. 仅验证 vLLM 版本兼容性修复")
        print(f"  {Colors.GREEN}4{Colors.RESET}. 仅验证配置文件状态")
        print(f"  {Colors.GREEN}5{Colors.RESET}. 仅验证内存优化状态")
        print(f"  {Colors.GREEN}0{Colors.RESET}. 返回主菜单")
        
        print(f"\n请输入选项 (0-5): ", end='')
        choice = input().strip()
        
        if choice == '1' or choice == '':
            self.verify_fixes(categories=['t4', 'vllm', 'config', 'memory'])
        elif choice == '2':
            self.verify_fixes(categories=['t4'])
        elif choice == '3':
            self.verify_fixes(categories=['vllm'])
        elif choice == '4':
            self.verify_fixes(categories=['config'])
        elif choice == '5':
            self.verify_fixes(categories=['memory'])
        elif choice == '0':
            return
        else:
            print(f"\n{Colors.RED}❌ 无效的选项{Colors.RESET}")
    
    def run_interactive(self):
        """交互式运行修复流程"""
        while True:
            choice = self.show_menu()
            
            if choice == '' or choice == '0':
                print(f"\n{Colors.CYAN}👋 退出程序{Colors.RESET}\n")
                break
            
            elif choice == '1':
                # 完整修复
                self.run_full_fix()
            
            elif choice == '2':
                # 仅 T4 GPU 修复
                self.run_t4_fix_only()
            
            elif choice == '3':
                # 仅 vLLM 兼容性修复
                self.run_vllm_fix_only()
            
            elif choice == '4':
                # 恢复备份
                self.restore_from_backup()
            
            elif choice == '5':
                # 验证状态
                if self.check_environment():
                    self.show_verify_menu()
            
            elif choice == '6':
                # 创建独立配置文件
                self.create_separate_configs()
            
            elif choice == '7':
                # 添加内存优化
                self.add_memory_optimization()
            
            else:
                print(f"\n{Colors.RED}❌ 无效的选项，请重新选择{Colors.RESET}")
            
            print(f"\n{Colors.CYAN}按回车键继续...{Colors.RESET}")
            input()
    
    def run_full_fix(self):
        """运行完整修复 (T4 + vLLM)"""
        self.reset_stats()
        
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}🔧 完整修复 (T4 GPU + vLLM 兼容性){Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        
        if not self.check_environment():
            return False
        
        if not self.create_backup():
            return False
        
        self.create_directories_and_update_config()
        
        # T4 GPU 修复
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔨 开始 T4 GPU 兼容性修复{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        
        for filename in self.files_to_fix:
            self.fix_t4_file(filename)
        
        # vLLM 兼容性修复
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔨 开始 vLLM 版本兼容性修复{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        
        for filename in self.files_to_fix:
            self.fix_vllm_imports(filename)
        
        self.verify_fixes()
        self.generate_report()
        
        return True
    
    def run_t4_fix_only(self):
        """仅运行 T4 GPU 修复"""
        self.reset_stats()
        
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}🔧 T4 GPU 兼容性修复{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        
        if not self.check_environment():
            return False
        
        if not self.create_backup():
            return False
        
        self.create_directories_and_update_config()
        
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔨 开始 T4 GPU 兼容性修复{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        
        for filename in self.files_to_fix:
            self.fix_t4_file(filename)
        
        self.verify_fixes()
        self.generate_report()
        
        return True
    
    def run_vllm_fix_only(self):
        """仅运行 vLLM 兼容性修复"""
        self.reset_stats()
        
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}🔧 vLLM 版本兼容性修复{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        
        if not self.check_environment():
            return False
        
        if not self.create_backup():
            return False
        
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔨 开始 vLLM 版本兼容性修复{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        
        for filename in self.files_to_fix:
            self.fix_vllm_imports(filename)
        
        self.generate_report()
        
        return True
    
    # ========================================================================
    # 创建独立配置文件功能
    # ========================================================================
    
    def create_separate_configs(self):
        """创建独立的配置文件（图片/PDF/批量）并更新脚本引用"""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}📁 创建独立配置文件{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        
        if not self.vllm_path.exists():
            print(f"\n{Colors.RED}❌ 错误: vLLM 路径不存在: {self.vllm_path}{Colors.RESET}")
            return False
        
        config_file = self.vllm_path / 'config.py'
        if not config_file.exists():
            print(f"\n{Colors.RED}❌ 错误: config.py 不存在{Colors.RESET}")
            return False
        
        # 创建备份（包括当前的配置文件和脚本文件）
        if not self.create_backup(include_configs=True):
            return False
        
        # 读取原始 config.py
        with open(config_file, 'r', encoding='utf-8') as f:
            original_config = f.read()
        
        print(f"\n{Colors.BLUE}📄 读取原始配置文件: config.py{Colors.RESET}")
        
        # 配置文件定义
        config_definitions = {
            'config_image.py': {
                'description': '单张图片处理配置',
                'input_dir': 'input_image',
                'output_dir': 'output_image',
                'script': 'run_dpsk_ocr_image.py',
                'header': '''"""
DeepSeek-OCR 单张图片处理配置文件
=================================

专用于 run_dpsk_ocr_image.py 脚本的配置

使用方法：
    在 run_dpsk_ocr_image.py 中导入：
    from config_image import *

作者：DeepSeek AI
修改日期：{date}
版本：v1.0
"""

import os
from pathlib import Path

''',
                'path_section': '''
# ============================================================================
# 输入输出路径 - 单张图片处理
# ============================================================================
# 获取当前脚本所在目录
CURRENT_DIR = Path(__file__).parent

# 输入：单张图片文件或图片目录
INPUT_DIR = CURRENT_DIR / '{input_dir}'
INPUT_PATH = str(INPUT_DIR)

# 如果想指定具体图片，取消下面的注释：
# INPUT_PATH = str(INPUT_DIR / 'test_image.jpg')

# 输出：结果保存目录
OUTPUT_DIR = CURRENT_DIR / '{output_dir}'
OUTPUT_PATH = str(OUTPUT_DIR)

# 自动创建目录
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / 'images', exist_ok=True)

'''
            },
            'config_pdf.py': {
                'description': 'PDF文档处理配置',
                'input_dir': 'input_pdf',
                'output_dir': 'output_pdf',
                'script': 'run_dpsk_ocr_pdf.py',
                'header': '''"""
DeepSeek-OCR PDF文档处理配置文件
================================

专用于 run_dpsk_ocr_pdf.py 脚本的配置

使用方法：
    在 run_dpsk_ocr_pdf.py 中导入：
    from config_pdf import *

作者：DeepSeek AI
修改日期：{date}
版本：v1.0
"""

import os
from pathlib import Path

''',
                'path_section': '''
# ============================================================================
# 输入输出路径 - PDF文档处理
# ============================================================================
# 获取当前脚本所在目录
CURRENT_DIR = Path(__file__).parent

# 输入：PDF文件路径
INPUT_DIR = CURRENT_DIR / '{input_dir}'
INPUT_PATH = str(INPUT_DIR)

# 如果是单个PDF，取消下面的注释并指定文件：
# INPUT_PATH = str(INPUT_DIR / 'document.pdf')

# 输出：结果保存目录
OUTPUT_DIR = CURRENT_DIR / '{output_dir}'
OUTPUT_PATH = str(OUTPUT_DIR)

# 自动创建目录
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

'''
            },
            'config_batch.py': {
                'description': '批量图片处理配置',
                'input_dir': 'input_batch',
                'output_dir': 'output_batch',
                'script': 'run_dpsk_ocr_eval_batch.py',
                'header': '''"""
DeepSeek-OCR 批量图片处理配置文件
=================================

专用于 run_dpsk_ocr_eval_batch.py 脚本的配置

使用方法：
    在 run_dpsk_ocr_eval_batch.py 中导入：
    from config_batch import *

作者：DeepSeek AI
修改日期：{date}
版本：v1.0
"""

import os
from pathlib import Path

''',
                'path_section': '''
# ============================================================================
# 输入输出路径 - 批量图片处理
# ============================================================================
# 获取当前脚本所在目录
CURRENT_DIR = Path(__file__).parent

# 输入：包含多张图片的文件夹
INPUT_DIR = CURRENT_DIR / '{input_dir}'
INPUT_PATH = str(INPUT_DIR)

# 输出：结果保存目录
OUTPUT_DIR = CURRENT_DIR / '{output_dir}'
OUTPUT_PATH = str(OUTPUT_DIR)

# 自动创建目录
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

'''
            },
            'config_pdf_batch.py': {
                'description': '批量PDF处理配置',
                'input_dir': 'input_pdf_batch',
                'output_dir': 'output_pdf_batch',
                'script': 'run_dpsk_ocr_pdf_batch.py',
                'header': '''"""
DeepSeek-OCR 批量PDF处理配置文件
================================

专用于 run_dpsk_ocr_pdf_batch.py 脚本的配置

使用方法：
    在 run_dpsk_ocr_pdf_batch.py 中导入：
    from config_pdf_batch import *

作者：DeepSeek AI
修改日期：{date}
版本：v1.0
"""

import os
from pathlib import Path

''',
                'path_section': '''
# ============================================================================
# 输入输出路径 - 批量PDF处理
# ============================================================================
# 获取当前脚本所在目录
CURRENT_DIR = Path(__file__).parent

# 输入：包含多个PDF文件的文件夹
INPUT_DIR = CURRENT_DIR / '{input_dir}'
INPUT_PATH = str(INPUT_DIR)

# 输出：结果保存目录（每个PDF会创建子目录）
OUTPUT_DIR = CURRENT_DIR / '{output_dir}'
OUTPUT_PATH = str(OUTPUT_DIR)

# 自动创建目录
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

'''
            }
        }
        
        # 从原始 config.py 提取核心配置部分
        core_config = self._extract_core_config(original_config)
        
        created_configs = []
        
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}📝 创建配置文件{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        for config_name, config_def in config_definitions.items():
            config_path = self.vllm_path / config_name
            
            # 生成配置文件内容
            date_str = datetime.now().strftime('%Y-%m-%d')
            header = config_def['header'].format(date=date_str)
            path_section = config_def['path_section'].format(
                input_dir=config_def['input_dir'],
                output_dir=config_def['output_dir']
            )
            
            # 组合配置文件
            config_content = header + core_config + path_section
            
            # 添加分词器初始化
            config_content += '''
# ============================================================================
# 分词器初始化
# ============================================================================
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
'''
            
            # 写入配置文件
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            # 创建目录
            input_dir = self.vllm_path / config_def['input_dir']
            output_dir = self.vllm_path / config_def['output_dir']
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"{Colors.GREEN}✓{Colors.RESET} 已创建: {config_name} ({config_def['description']})")
            print(f"  输入目录: {config_def['input_dir']}/")
            print(f"  输出目录: {config_def['output_dir']}/")
            
            created_configs.append({
                'config': config_name,
                'script': config_def['script'],
                'description': config_def['description']
            })
        
        # 询问是否更新脚本引用
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔗 更新脚本配置引用{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        print(f"{Colors.YELLOW}是否更新运行脚本的配置引用？{Colors.RESET}")
        print(f"  这将修改以下脚本：")
        for item in created_configs:
            print(f"    • {item['script']} → from {item['config'].replace('.py', '')} import ...")
        
        print(f"\n输入 'y' 确认更新，其他输入跳过: ", end='')
        confirm = input().strip().lower()
        
        if confirm == 'y':
            self._update_script_config_imports(created_configs)
        else:
            print(f"{Colors.YELLOW}⊘{Colors.RESET} 跳过更新脚本引用")
            print(f"\n{Colors.CYAN}💡 手动更新方法:{Colors.RESET}")
            for item in created_configs:
                config_module = item['config'].replace('.py', '')
                print(f"  在 {item['script']} 中将:")
                print(f"    from config import ... → from {config_module} import ...")
        
        print(f"\n{Colors.GREEN}✅ 配置文件创建完成！{Colors.RESET}")
        return True
    
    def _extract_core_config(self, config_content):
        """从原始 config.py 提取核心配置部分"""
        # 提取模型规格配置到提示词配置之间的内容
        core_sections = []
        
        # 提取模型规格配置
        model_spec_pattern = r'(# ={70,}\n# 模型规格配置.*?(?=# ={70,}\n# 输入输出路径|# ={70,}\n# 分词器))'
        model_spec_match = re.search(model_spec_pattern, config_content, re.DOTALL)
        if model_spec_match:
            core_sections.append(model_spec_match.group(1))
        else:
            # 备用：提取基本变量
            core_sections.append('''# ============================================================================
# 模型规格配置
# ============================================================================
BASE_SIZE = 1024      # 基础图像大小（全局视图）
IMAGE_SIZE = 640      # 裁剪图像大小（局部视图）
CROP_MODE = True      # 是否启用图像裁剪模式

# ============================================================================
# 裁剪配置
# ============================================================================
MIN_CROPS = 2         # 最小裁剪数量
MAX_CROPS = 6         # 最大裁剪数量

# ============================================================================
# 并发和性能配置
# ============================================================================
MAX_CONCURRENCY = 100 # 最大并发处理数量
NUM_WORKERS = 64      # 图像预处理工作线程数

# ============================================================================
# 调试和输出配置
# ============================================================================
PRINT_NUM_VIS_TOKENS = False  # 是否打印视觉 token 数量
SKIP_REPEAT = True            # 是否跳过重复内容

# ============================================================================
# 模型路径配置
# ============================================================================
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'

# ============================================================================
# 提示词配置
# ============================================================================
PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'

''')
        
        return ''.join(core_sections)
    
    def _update_script_config_imports(self, config_mappings):
        """更新运行脚本和共享模块中的配置导入"""
        
        # 主脚本到配置文件的映射
        script_import_map = {
            'run_dpsk_ocr_image.py': 'config_image',
            'run_dpsk_ocr_pdf.py': 'config_pdf',
            'run_dpsk_ocr_eval_batch.py': 'config_batch',
            'run_dpsk_ocr_pdf_batch.py': 'config_pdf_batch',
        }
        
        # 共享模块列表（这些模块也需要更新导入）
        shared_modules = [
            'deepseek_ocr.py',
            'process/image_process.py',
        ]
        
        print(f"\n{Colors.BLUE}更新主脚本配置导入...{Colors.RESET}")
        
        for script_name, config_module in script_import_map.items():
            script_path = self.vllm_path / script_name
            
            if not script_path.exists():
                print(f"  {Colors.YELLOW}⊘{Colors.RESET} 脚本不存在: {script_name}")
                continue
            
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换导入语句
            pattern = r'from\s+config\s+import\s+'
            if re.search(pattern, content) and f'from {config_module} import' not in content:
                new_content = re.sub(pattern, f'from {config_module} import ', content)
                
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  {Colors.GREEN}✓{Colors.RESET} {script_name} → {config_module}")
            else:
                print(f"  {Colors.YELLOW}⊘{Colors.RESET} {script_name} 已是最新或无需更新")
        
        # 询问是否更新共享模块
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.YELLOW}⚠️  关于共享模块的重要说明：{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"""
共享模块 (deepseek_ocr.py, process/image_process.py) 被所有脚本共同使用。
如果将它们改为使用特定配置文件，其他脚本将无法正常工作。

{Colors.GREEN}推荐方案：{Colors.RESET}
  • 保持共享模块使用 config.py
  • 只需确保 config.py 中的 MODEL_PATH 正确即可
  • 各专用配置文件主要用于设置不同的输入/输出路径

{Colors.YELLOW}如果您确实需要为每个脚本使用完全独立的配置：{Colors.RESET}
  • 需要为每个脚本创建独立的共享模块副本
  • 这会增加维护复杂度

是否同步更新共享模块？(不推荐)
  输入 'shared' 更新共享模块为使用当前选择的配置
  输入其他任意内容跳过（推荐）
""")
        print(f"请输入: ", end='')
        user_input = input().strip().lower()
        
        if user_input == 'shared':
            print(f"\n{Colors.YELLOW}⚠️  警告：这将影响所有脚本的运行！{Colors.RESET}")
            print(f"请选择要使用的配置文件：")
            print(f"  1. config_image (单图片)")
            print(f"  2. config_pdf (PDF)")
            print(f"  3. config_batch (批量图片)")
            print(f"  4. config_pdf_batch (批量PDF)")
            print(f"\n输入数字 (1-4): ", end='')
            
            config_choice = input().strip()
            config_map = {
                '1': 'config_image',
                '2': 'config_pdf',
                '3': 'config_batch',
                '4': 'config_pdf_batch'
            }
            
            if config_choice in config_map:
                target_config = config_map[config_choice]
                self._update_shared_modules(shared_modules, target_config)
            else:
                print(f"{Colors.YELLOW}⊘{Colors.RESET} 无效选择，跳过更新共享模块")
        else:
            print(f"{Colors.GREEN}✓{Colors.RESET} 保持共享模块使用 config.py（推荐）")
            print(f"\n{Colors.CYAN}💡 提示：请确保 config.py 中的 MODEL_PATH 设置正确{Colors.RESET}")
    
    def _update_shared_modules(self, modules, target_config):
        """更新共享模块的配置导入"""
        print(f"\n{Colors.BLUE}更新共享模块配置导入 → {target_config}...{Colors.RESET}")
        
        for module_path in modules:
            full_path = self.vllm_path / module_path
            
            if not full_path.exists():
                print(f"  {Colors.YELLOW}⊘{Colors.RESET} 模块不存在: {module_path}")
                continue
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 备份原文件
            backup_path = str(full_path) + '.backup_shared'
            if not os.path.exists(backup_path):
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # 替换导入语句
            pattern = r'from\s+config\s+import\s+'
            if re.search(pattern, content):
                new_content = re.sub(pattern, f'from {target_config} import ', content)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  {Colors.GREEN}✓{Colors.RESET} {module_path} → {target_config}")
            else:
                print(f"  {Colors.YELLOW}⊘{Colors.RESET} {module_path} 未找到 config 导入")
        
        print(f"\n{Colors.YELLOW}⚠️  注意：共享模块已更新！{Colors.RESET}")
        print(f"  如需恢复，请使用功能 4（恢复备份）或手动恢复 .backup_shared 文件")
    
    # ========================================================================
    # 内存优化功能
    # ========================================================================
    
    def add_memory_optimization(self):
        """添加内存优化代码，防止批量处理时OOM"""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}🧠 添加内存优化{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}内存优化内容：{Colors.RESET}")
        print(f"  1. 添加 gc.collect() 垃圾回收")
        print(f"  2. 添加 torch.cuda.empty_cache() GPU缓存清理")
        print(f"  3. 处理完每个文件后释放内存")
        print(f"  4. 删除不再使用的大型变量")
        print(f"  5. 批量处理改为分批处理（可选）")
        
        if not self.check_environment():
            return False
        
        if not self.create_backup(include_configs=True):
            return False
        
        print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🔨 开始添加内存优化{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        
        scripts_to_optimize = [
            ('run_dpsk_ocr_pdf_batch.py', self._add_memory_opt_pdf_batch),
            ('run_dpsk_ocr_eval_batch.py', self._add_memory_opt_eval_batch),
            ('run_dpsk_ocr_pdf.py', self._add_memory_opt_pdf),
            ('run_dpsk_ocr_image.py', self._add_memory_opt_image),
        ]
        
        for script_name, optimize_func in scripts_to_optimize:
            script_path = self.vllm_path / script_name
            if script_path.exists():
                print(f"\n{Colors.BLUE}📝 优化: {script_name}{Colors.RESET}")
                try:
                    fixes = optimize_func(script_path)
                    if fixes:
                        print(f"  {Colors.GREEN}✓{Colors.RESET} 已添加 {len(fixes)} 处内存优化:")
                        for fix in fixes:
                            print(f"      • {fix}")
                    else:
                        print(f"  {Colors.YELLOW}⊘{Colors.RESET} 已包含内存优化或无需修改")
                except Exception as e:
                    print(f"  {Colors.RED}✗{Colors.RESET} 优化失败: {e}")
            else:
                print(f"\n{Colors.YELLOW}⊘{Colors.RESET} 脚本不存在: {script_name}")
        
        print(f"\n{Colors.GREEN}✅ 内存优化添加完成！{Colors.RESET}")
        return True
    
    def _add_memory_opt_pdf_batch(self, filepath):
        """为 run_dpsk_ocr_pdf_batch.py 添加内存优化（重点优化RAM）"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 添加 gc 导入
        if 'import gc' not in content:
            content = content.replace(
                'import torch\n',
                'import torch\nimport gc\n'
            )
            fixes.append('添加 gc 模块导入')
        
        # 2. 添加内存清理函数
        memory_cleanup_func = '''
def cleanup_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 创建全局单例处理器（避免重复创建导致内存泄漏）
_global_processor = None
def get_processor():
    global _global_processor
    if _global_processor is None:
        _global_processor = DeepseekOCRProcessor()
    return _global_processor

'''
        if 'def cleanup_memory():' not in content:
            content = content.replace(
                'class Colors:',
                memory_cleanup_func + 'class Colors:'
            )
            fixes.append('添加 cleanup_memory() 和全局处理器单例')
        
        # 3. 修复关键问题：process_single_image 中每次创建新的处理器实例
        old_process_image = '''def process_single_image(image):
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
    return cache_item'''
        
        new_process_image = '''def process_single_image(image):
    """
    预处理单张图片（多线程版本，内存优化）
    
    Args:
        image (Image): PIL Image 对象
        
    Returns:
        dict: 包含提示词和图像特征的字典
    """
    prompt_in = PROMPT
    # 使用全局单例处理器，避免每次创建新实例导致内存泄漏
    processor = get_processor()
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {
            "image": processor.tokenize_with_images(
                images=[image], 
                bos=True, 
                eos=True, 
                cropping=CROP_MODE
            )
        },
    }
    return cache_item'''
        
        if old_process_image in content:
            content = content.replace(old_process_image, new_process_image)
            fixes.append('使用全局单例处理器（关键内存优化）')
        
        # 4. 在 process_single_pdf 函数中添加分批处理和清理
        # 这是最关键的内存优化：将所有页面分批处理，每批处理完后释放内存
        
        # 模式1：原始未修改的格式
        old_batch_process_v1 = '''        # 2. 多线程预处理
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
        )'''
        
        # 模式2：已经被限制线程数修改过的格式
        old_batch_process_v2 = '''        # 2. 多线程预处理
        print(f"{Colors.BLUE}🔄 正在预处理图片...{Colors.RESET}")
        # 注意：NUM_WORKERS 过高会导致内存占用过大，建议设置为 4-8
        with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, 8)) as executor:  
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
        )'''
        
        new_batch_process = '''        # 2. 多线程预处理（分批处理以节省内存）
        print(f"{Colors.BLUE}🔄 正在预处理图片...{Colors.RESET}")
        
        # 分批处理配置 - 每批处理的页面数量
        PAGE_BATCH_SIZE = 20  # 可根据RAM大小调整：16GB RAM建议10，32GB建议20，64GB+建议30
        
        outputs_list = []
        total_batches = (len(images) + PAGE_BATCH_SIZE - 1) // PAGE_BATCH_SIZE
        
        for batch_idx in range(0, len(images), PAGE_BATCH_SIZE):
            batch_images = images[batch_idx:batch_idx + PAGE_BATCH_SIZE]
            current_batch = batch_idx // PAGE_BATCH_SIZE + 1
            print(f"  📦 处理批次 {current_batch}/{total_batches} ({len(batch_images)} 页)...")
            
            # 预处理当前批次
            # 注意：NUM_WORKERS 过高会导致内存占用过大，建议设置为 4-8
            with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, 8)) as executor:
                batch_inputs = list(tqdm(
                    executor.map(process_single_image, batch_images),
                    total=len(batch_images),
                    desc=f"批次 {current_batch}",
                    colour='blue',
                    leave=False
                ))
            
            # OCR推理当前批次
            batch_outputs = llm.generate(
                batch_inputs,
                sampling_params=sampling_params
            )
            outputs_list.extend(batch_outputs)
            
            # 立即释放当前批次的内存
            del batch_images, batch_inputs, batch_outputs
            cleanup_memory()
            print(f"  ✓ 批次 {current_batch} 完成，已释放内存")
        
        print(f"{Colors.GREEN}✓{Colors.RESET} OCR识别完成")'''
        
        if 'PAGE_BATCH_SIZE' not in content:
            # 尝试匹配模式1（原始格式）
            if old_batch_process_v1 in content:
                content = content.replace(old_batch_process_v1, new_batch_process)
                fixes.append('添加分批处理逻辑（关键RAM优化）')
            # 尝试匹配模式2（已限制线程数格式）
            elif old_batch_process_v2 in content:
                content = content.replace(old_batch_process_v2, new_batch_process)
                fixes.append('添加分批处理逻辑（关键RAM优化）')
            else:
                # 如果两种模式都不匹配，至少限制线程数
                old_executor = 'with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:'
                new_executor = '''# 注意：NUM_WORKERS 过高会导致内存占用过大，建议设置为 4-8
        with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, 8)) as executor:'''
                
                if old_executor in content and '# 注意：NUM_WORKERS 过高' not in content:
                    content = content.replace(old_executor, new_executor)
                    fixes.append('限制最大线程数为8（防止RAM溢出）')
        
        # 6. 在处理完成后添加清理（在 return 之后进行，避免影响 return 语句中的变量引用）
        # 注意：不在 return 之前删除 images，因为 return 语句需要 len(images)
        # 内存清理将在主循环中进行
        
        # 7. 在主循环每个PDF后清理
        old_loop = 'result = process_single_pdf(pdf_file, OUTPUT_PATH)'
        new_loop = '''result = process_single_pdf(pdf_file, OUTPUT_PATH)
        
        # 每处理完一个PDF就强制清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()'''
        
        if old_loop in content and '# 每处理完一个PDF就强制清理内存' not in content:
            content = content.replace(old_loop, new_loop)
            fixes.append('在PDF间添加强制内存清理')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def _add_memory_opt_eval_batch(self, filepath):
        """为 run_dpsk_ocr_eval_batch.py 添加内存优化（重点优化RAM）"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 添加 gc 导入
        if 'import gc' not in content:
            content = content.replace(
                'import torch\n',
                'import torch\nimport gc\n'
            )
            fixes.append('添加 gc 模块导入')
        
        # 2. 添加内存清理函数和全局处理器单例
        memory_cleanup_func = '''
def cleanup_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 创建全局单例处理器（避免重复创建导致内存泄漏）
_global_processor = None
def get_processor():
    global _global_processor
    if _global_processor is None:
        _global_processor = DeepseekOCRProcessor()
    return _global_processor

'''
        if 'def cleanup_memory():' not in content:
            content = content.replace(
                'class Colors:',
                memory_cleanup_func + 'class Colors:'
            )
            fixes.append('添加 cleanup_memory() 和全局处理器单例')
        
        # 3. 修复 process_single_image 中创建新处理器的问题
        old_process = 'DeepseekOCRProcessor().tokenize_with_images('
        new_process = 'get_processor().tokenize_with_images('
        
        if old_process in content and 'get_processor()' not in content:
            content = content.replace(old_process, new_process)
            fixes.append('使用全局单例处理器（关键内存优化）')
        
        # 4. 完全重写批量处理逻辑 - 分批加载和处理
        # 匹配原始文件中的实际代码模式
        old_batch_section_v1 = '''    images = []

    for image_path in images_path:
        image = Image.open(image_path).convert('RGB')
        images.append(image)

    prompt = PROMPT

    # batch_inputs = []


    # for image in tqdm(images):

    #     prompt_in = prompt
    #     cache_list = [
    #         {
    #             "prompt": prompt_in,
    #             "multi_modal_data": {"image": Image.open(image).convert('RGB')},
    #         }
    #     ]
    #     batch_inputs.extend(cache_list)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc="Pre-processed images"
        ))


    

    outputs_list = llm.generate(
        batch_inputs,
        sampling_params=sampling_params
    )'''
        
        # 匹配简化版本
        old_batch_section_v2 = '''    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc="Pre-processed images"
        ))


    

    outputs_list = llm.generate(
        batch_inputs,
        sampling_params=sampling_params
    )'''
        
        new_batch_section = '''    prompt = PROMPT
    
    # 分批处理配置 - 根据RAM大小调整
    BATCH_SIZE = 10  # 每批处理的图片数量：16GB RAM建议5，32GB建议10，64GB+建议20
    
    outputs_list = []
    total_batches = (len(images_path) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f'{Colors.GREEN}开始分批处理 ({total_batches} 批次，每批 {BATCH_SIZE} 张)...{Colors.RESET}')
    
    for batch_idx in range(0, len(images_path), BATCH_SIZE):
        batch_paths = images_path[batch_idx:batch_idx + BATCH_SIZE]
        current_batch = batch_idx // BATCH_SIZE + 1
        print(f'\\n  📦 批次 {current_batch}/{total_batches}')
        
        # 加载当前批次的图片
        batch_images = []
        for img_path in batch_paths:
            img = Image.open(img_path).convert('RGB')
            batch_images.append(img)
        
        # 预处理当前批次（限制线程数防止内存溢出）
        with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, 4)) as executor:
            batch_inputs = list(tqdm(
                executor.map(process_single_image, batch_images),
                total=len(batch_images),
                desc=f"预处理批次 {current_batch}",
                leave=False
            ))
        
        # 释放原始图片
        del batch_images
        gc.collect()
        
        # OCR推理当前批次
        batch_outputs = llm.generate(
            batch_inputs,
            sampling_params=sampling_params
        )
        outputs_list.extend(batch_outputs)
        
        # 释放当前批次数据
        del batch_inputs, batch_outputs
        cleanup_memory()
        print(f'  ✓ 批次 {current_batch} 完成')'''
        
        if 'BATCH_SIZE' not in content:
            if old_batch_section_v1 in content:
                content = content.replace(old_batch_section_v1, new_batch_section)
                fixes.append('完全重写为分批加载处理（关键RAM优化）')
            elif old_batch_section_v2 in content:
                content = content.replace(old_batch_section_v2, new_batch_section)
                fixes.append('完全重写为分批加载处理（关键RAM优化）')
        
        # 5. 添加最终清理 - 在文件末尾添加清理代码
        old_end = '''        mmd_path = output_path + image.split('/')[-1].replace('.jpg', '.md')

        with open(mmd_path, 'w', encoding='utf-8') as afile:
            afile.write(content)'''
        
        new_end = '''        mmd_path = output_path + image.split('/')[-1].replace('.jpg', '.md')

        with open(mmd_path, 'w', encoding='utf-8') as afile:
            afile.write(content)
    
    # 最终内存清理
    del outputs_list
    cleanup_memory()
    print(f'{Colors.GREEN}批量处理完成！共处理 {len(images_path)} 张图片{Colors.RESET}')'''
        
        if old_end in content and '# 最终内存清理' not in content:
            content = content.replace(old_end, new_end)
            fixes.append('添加最终内存清理')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def _add_memory_opt_pdf(self, filepath):
        """为 run_dpsk_ocr_pdf.py 添加内存优化（重点优化RAM）"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 添加 gc 导入
        if 'import gc' not in content:
            content = content.replace(
                'import torch\n',
                'import torch\nimport gc\n'
            )
            fixes.append('添加 gc 模块导入')
        
        # 2. 添加内存清理函数和全局处理器
        memory_cleanup_func = '''
def cleanup_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 创建全局单例处理器（避免重复创建导致内存泄漏）
_global_processor = None
def get_processor():
    global _global_processor
    if _global_processor is None:
        _global_processor = DeepseekOCRProcessor()
    return _global_processor

'''
        if 'def cleanup_memory():' not in content:
            content = content.replace(
                'class Colors:',
                memory_cleanup_func + 'class Colors:'
            )
            fixes.append('添加 cleanup_memory() 和全局处理器单例')
        
        # 3. 修复 process_single_image 中创建新处理器的问题
        old_process = 'DeepseekOCRProcessor().tokenize_with_images('
        new_process = 'get_processor().tokenize_with_images('
        
        if old_process in content:
            content = content.replace(old_process, new_process)
            fixes.append('使用全局单例处理器（关键内存优化）')
        
        # 4. 限制线程数
        old_executor = 'with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:'
        new_executor = '''# 注意：NUM_WORKERS 过高会导致内存占用过大，建议设置为 4-8
        with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, 8)) as executor:'''
        
        if old_executor in content and '# 注意：NUM_WORKERS 过高' not in content:
            content = content.replace(old_executor, new_executor)
            fixes.append('限制最大线程数为8（防止RAM溢出）')
        
        # 5. 在处理完成后清理
        old_success = "print(f'{Colors.GREEN}✅ 处理完成！{Colors.RESET}')"
        new_success = '''# 最终内存清理
        try:
            del images, draw_images
        except:
            pass
        cleanup_memory()
        
        print(f'{Colors.GREEN}✅ 处理完成！{Colors.RESET}')'''
        
        if old_success in content and '# 最终内存清理' not in content:
            content = content.replace(old_success, new_success)
            fixes.append('添加最终内存清理')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def _add_memory_opt_image(self, filepath):
        """为 run_dpsk_ocr_image.py 添加内存优化（重点优化RAM）"""
        fixes = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 添加 gc 导入
        if 'import gc' not in content:
            content = content.replace(
                'import torch\n',
                'import torch\nimport gc\n'
            )
            fixes.append('添加 gc 模块导入')
        
        # 2. 添加内存清理函数
        memory_cleanup_func = '''
def cleanup_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

'''
        if 'def cleanup_memory():' not in content:
            # 找到合适的位置添加
            if 'class Colors:' in content:
                content = content.replace(
                    'class Colors:',
                    memory_cleanup_func + 'class Colors:'
                )
                fixes.append('添加 cleanup_memory() 函数')
            elif 'def load_image' in content:
                content = content.replace(
                    'def load_image',
                    memory_cleanup_func + 'def load_image'
                )
                fixes.append('添加 cleanup_memory() 函数')
        
        # 3. 在处理完成后添加清理
        if "if __name__ ==" in content and 'cleanup_memory()' not in content:
            # 找到 main 函数的末尾，添加清理
            old_main = 'if __name__ == "__main__":'
            new_main = '''# 处理完成后释放内存
def cleanup_after_processing():
    cleanup_memory()
    print("内存已清理")

if __name__ == "__main__":'''
            
            if old_main in content and 'cleanup_after_processing' not in content:
                content = content.replace(old_main, new_main)
                fixes.append('添加处理后清理函数')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixes
        return None
    
    def run(self):
        """
        运行修复流程（默认完整修复，保持向后兼容）
        """
        return self.run_full_fix()


def main():
    """主函数"""
    # 解析命令行参数
    project_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 创建修复器
    fixer = T4CompatibilityFixer(project_path)
    
    # 检查是否有命令行参数指定非交互模式
    if len(sys.argv) > 2 and sys.argv[2] == '--auto':
        # 自动模式：完整修复
        success = fixer.run()
        sys.exit(0 if success else 1)
    else:
        # 交互模式
        fixer.run_interactive()


if __name__ == '__main__':
    main()
