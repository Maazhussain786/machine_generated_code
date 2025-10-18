
import argparse
from data_loader import load_task_parquet
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["tfidf","xgboost","codebert"])
    p.add_argument("--task", required=True, choices=["task_a","task_b","task_c"])
    p.add_argument("--data_root", default=r"D:\Projects\Final_project\data")
    p.add_argument("--out_dir", default="experiments")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    args = p.parse_args()

    train_df, dev_df, test_df = load_task_parquet(args.data_root, args.task)

    if args.model == "tfidf":
        from model_tfidf import train_tfidf
        out = os.path.join(args.out_dir, f"tfidf_{args.task}")
        train_tfidf(train_df, dev_df, out_dir=out)
    elif args.model == "xgboost":
        from model_xgboost import train_xgboost
        out = os.path.join(args.out_dir, f"xgboost_{args.task}")
        train_xgboost(train_df, dev_df, out_dir=out)
    else:
        from model_codebert import train_codebert
        out = os.path.join(args.out_dir, f"codebert_{args.task}")
        train_codebert(train_df, dev_df, out_dir=out, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length)

if __name__ == "__main__":
    main()
