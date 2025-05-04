import os,sys

# 取得目前的工作目錄 (預期是 tim-main)
current_dir = os.getcwd()
print("Current Working Directory:", current_dir)

# 設定 recognition 的路徑 (假設 recognition 位於 tim-main 底下)
recognition_dir = os. path.join(current_dir, "recognition")
os.chdir(recognition_dir)

print("New Working Directory:", os.getcwd())

sys.path.append(os.getcwd())
print("sys.path:", sys.path)



from time_interval_machine.utils.parser import parse_args
from time_interval_machine.utils.misc import launch_job
from scripts.extract_feats import init_extract
from scripts.train import init_train
from scripts.test import init_test

def main():    
    args = parse_args()

    if args.train:
        launch_job(args=args, init_method=args.init_method, func=init_train)
    elif args.validate:
        launch_job(args=args, init_method=args.init_method, func=init_test)
    elif args.extract_feats:
        launch_job(args=args, init_method=args.init_method, func=init_extract)
    else:
        print("No script specified, please use [--train, --validate, --extract_feats]")


if __name__ == "__main__":
    main()
