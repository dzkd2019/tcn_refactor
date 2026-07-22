"""命令行入口，将生成、训练和端到端回放拆成互不产生副作用的子命令。"""

from __future__ import annotations

import argparse

from .config import SystemConfig
from .dataset import StreamWindowDataset
from .metrics import replay
from .session import StreamSession
from .synthetic import generate_scenarios, save_scenarios
from .training import evaluate_time_points, load_stream_model, train_stream_model
from torch.utils.data import DataLoader


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="气体传感器流式 TCN 工具")
    command = parser.add_subparsers(dest="command", required=True)

    generate = command.add_parser("generate", help="生成带事件真值的合成场景")
    generate.add_argument("--output", required=True, help="输出 .npz 路径")
    generate.add_argument("--count", type=int, default=12, help="场景数量")
    generate.add_argument("--duration-s", type=float, default=180.0, help="每个场景时长（秒）")
    generate.add_argument("--seed", type=int, default=2026, help="随机种子")
    generate.add_argument("--events", type=int, default=1,
                          help="每场景注入次数；正式训练建议为 1，避免未恢复残留")

    train = command.add_parser("train", help="训练仅流式的因果 TCN")
    train.add_argument("--data", required=True, help="输入场景 .npz 路径")
    train.add_argument("--val-data", help="独立验证场景 .npz 路径；正式训练建议提供")
    train.add_argument("--checkpoint", required=True, help="模型 checkpoint 输出路径")
    train.add_argument("--epochs", type=int, default=20, help="训练轮数")
    train.add_argument("--batch-size", type=int, default=32, help="每批窗口数")
    train.add_argument("--lr", type=float, default=5e-4, help="AdamW 学习率；续训时建议降低")
    train.add_argument("--device", default="cpu", help="例如 cpu 或 cuda")
    train.add_argument("--resume", help="从已有最佳 checkpoint 续训")

    evaluate = command.add_parser("replay", help="逐点回放，评估完整部署路径")
    evaluate.add_argument("--data", required=True, help="输入场景 .npz 路径")
    evaluate.add_argument("--checkpoint", required=True, help="已训练模型路径")
    evaluate.add_argument("--device", default="cpu", help="例如 cpu 或 cuda")
    evaluate.add_argument("--start", type=int, default=0, help="从第几个场景开始回放")
    evaluate.add_argument("--limit", type=int, help="本次最多回放多少个场景")

    window_eval = command.add_parser("window-eval", help="在已对齐独立测试窗口上评估模型")
    window_eval.add_argument("--data", required=True, help="测试场景 .npz 路径")
    window_eval.add_argument("--checkpoint", required=True, help="已训练模型路径")
    window_eval.add_argument("--batch-size", type=int, default=8, help="评估 batch 大小")
    window_eval.add_argument("--device", default="cpu", help="例如 cpu 或 cuda")
    return parser


def main() -> None:
    """解析命令行参数并调用相应工作流。"""
    args = _parser().parse_args()
    if args.command == "generate":
        config = SystemConfig()
        scenarios = generate_scenarios(args.count, config=config, duration_s=args.duration_s,
                                       seed=args.seed, events_per_scenario=args.events)
        save_scenarios(args.output, scenarios, config)
        print(f"已生成 {len(scenarios)} 个场景：{args.output}")
    elif args.command == "train":
        result = train_stream_model(args.data, args.checkpoint, validation_path=args.val_data, epochs=args.epochs,
                                    batch_size=args.batch_size, device=args.device, learning_rate=args.lr,
                                    resume_path=args.resume)
        print(f"训练完成：验证集 MAPE={result['mape']:.2f}%")
    elif args.command == "replay":
        model, config = load_stream_model(args.checkpoint, device=args.device)
        result = replay(args.data, StreamSession(model, config=config, device=args.device),
                        start=args.start, limit=args.limit)
        print("端到端回放：" + "，".join(f"{key}={value:.3f}" for key, value in result.items()))
    else:
        model, config = load_stream_model(args.checkpoint, device=args.device)
        dataset = StreamWindowDataset(args.data, config=config)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        result = evaluate_time_points(model, loader, model.head[-1].weight.device, config)
        print("独立窗口评估：" + "，".join(f"{key}={value:.3f}" for key, value in result.items()))
