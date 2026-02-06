# inspect_model.py
import torch
import os

# ====== 请在这里修改你的 .pth 文件名 ======
MODEL_PATH = "final_multilabel_edl.pth"
# =========================================

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 文件 '{MODEL_PATH}' 不存在，请检查路径！")
        return

    print(f"🔍 正在加载模型文件: {MODEL_PATH}")
    try:
        # 安全加载（显式关闭 weights_only 警告）
        data = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"✅ 成功加载！数据类型: {type(data)}\n")

    # 情况 1: 是字典（最常见：state_dict 或自定义 checkpoint）
    if isinstance(data, dict):
        print("📌 这是一个字典（dict），可能包含以下内容：")
        keys = list(data.keys())
        print(f"   键名列表: {keys}\n")

        # 检查是否是标准 state_dict（含 bert 层）
        if any('bert' in k for k in keys):
            print("✅ 检测到 BERT 权重，可能是标准 state_dict")
            # 尝试找分类层
            classifier_keys = [k for k in keys if ('classifier' in k or 'fc' in k or 'evidence' in k) and 'weight' in k]
            if classifier_keys:
                key = classifier_keys[0]
                shape = data[key].shape
                print(f"🎯 分类层 '{key}' 形状: {shape}")
                if len(shape) == 2:
                    num_labels = shape[0]
                    print(f"   → 推断标签数量: {num_labels}")
            else:
                print("⚠️ 未找到明显分类层，请手动检查键名")

        # 检查是否是自定义 checkpoint（如你之前的格式）
        if "num_classes" in keys and "model_states" in keys:
            print("✅ 检测到自定义 checkpoint 格式（多二分类器结构）")
            num_classes = data["num_classes"]
            model_states = data["model_states"]
            print(f"   - 标签数量: {num_classes}")
            print(f"   - 模型状态数量: {len(model_states)}")
            if isinstance(model_states, list) and len(model_states) > 0:
                first_state = model_states[0]
                if isinstance(first_state, dict):
                    print(f"   - 第一个模型的参数数量: {len(first_state)}")
                    # 尝试找 evidence_layer
                    evidence_keys = [k for k in first_state.keys() if 'evidence_layer' in k]
                    if evidence_keys:
                        print(f"   - 检测到 EDL 结构: {evidence_keys}")

    # 情况 2: 是列表（如 [state_dict, label_list]）
    elif isinstance(data, list):
        print(f"📌 这是一个列表，长度: {len(data)}")
        for i, item in enumerate(data[:3]):  # 只看前3项
            print(f"   第 {i+1} 项类型: {type(item)}")
            if isinstance(item, dict) and len(item) > 0:
                sample_key = next(iter(item))
                print(f"     示例键: {sample_key}, 值形状: {item[sample_key].shape if hasattr(item[sample_key], 'shape') else 'N/A'}")

    # 情况 3: 是完整模型对象
    elif hasattr(data, 'state_dict'):
        print("✅ 这是一个完整模型对象")
        state_dict = data.state_dict()
        print(f"   参数数量: {len(state_dict)}")
        # 找分类层
        for name, param in state_dict.items():
            if 'classifier' in name and 'weight' in name:
                print(f"   分类层 '{name}' 形状: {param.shape}")

    else:
        print("❓ 未知格式，请手动分析")

    print("\n" + "="*50)
    print("💡 下一步建议:")
    if isinstance(data, dict) and "num_classes" in data and "model_states" in data:
        print("  你的是「多二分类器 EDL 模型」，请使用你之前提供的推理逻辑。")
    elif isinstance(data, dict) and any('bert' in k for k in data.keys()):
        print("  你的是「标准 BERT 多标签模型」，可用通用分类代码。")
    else:
        print("  请根据上述信息定制加载逻辑。")

if __name__ == "__main__":
    main()