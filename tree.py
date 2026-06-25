import os

def print_tree(dir_path, padding='', exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = ['.venv', 'qwen_mcp_lora_output', 'qwen_merged']
        
    try:
        files = os.listdir(dir_path)
    except PermissionError:
        return

    # 过滤掉隐藏文件或特定不需要的处理
    files = [f for f in files if not f.startswith('~$')]
    
    count = len(files)
    for i, filename in enumerate(files):
        path = os.path.join(dir_path, filename)
        is_last = (i == count - 1)
        
        # 打印当前节点
        print(f"{padding}{'└── ' if is_last else '├── '}{filename}")
        
        # 如果是文件夹，且不在排除名单内，则继续递归展开
        if os.path.isdir(path):
            if filename in exclude_dirs:
                continue  # 匹配到不展开的文件夹，直接跳过其内部内容
            
            new_padding = padding + ('    ' if is_last else '│   ')
            print_tree(path, new_padding, exclude_dirs)

if __name__ == "__main__":
    print(".")
    print_tree(".")