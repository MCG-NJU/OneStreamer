from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'run_script'))
from eval_queue import main
if __name__=='__main__':main()
