# check_model.py
import torch

MODEL_PATH = "final_multilabel_edl.pth"

try:
    # 显式关闭警告
    data = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    
    print(f"✅ 成功加载！类型: {type(data)}")
    
    if isinstance(data, list):
        print(f"📌 这是一个列表，长度: {len(data)}")
        for i, item in enumerate(data):
            print(f"  第 {i+1} 项类型: {type(item)}")
            if isinstance(item, dict):
                print(f"    包含键名 (前5个): {list(item.keys())[:5]}")
                # 尝试找分类层
                for key in item.keys():
                    if 'weight' in key and ('classifier' in key or 'fc' in key):
                        print(f"    🎯 找到分类层: {key} → shape: {item[key].shape}")
            elif isinstance(item, list):
                print(f"    内容示例: {item[:3]}")
            else:
                print(f"    内容: {item}")
                
    elif isinstance(data, dict):
        print("✅ 是 state_dict")
        # 之前的逻辑...
        
    else:
        print(f"✅ 是完整模型对象: {type(data)}")

except Exception as e:
    print(f"❌ 加载失败: {e}")