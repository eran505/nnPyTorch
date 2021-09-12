
import os

def get_info_path_gen():
    path_dir = '{}/car_model/debug'.format(os.path.expanduser('~'))
    path_code = "/home/eranhe/eran/repo/Pursuit_Evasion/src/Attacker/PathGenrator.hpp"
    target_line=None
    with open(path_code,'r') as f :
        lines = f.readlines()
        for idx,line in enumerate(lines):
            if str(line).__contains__("py1"):
                print("yes")
                target_line = lines[idx+1]
                break
    with open("{}/t.txt".format(path_dir),'w') as f:
        f.write(target_line)
    print("done")

if __name__ == '__main__':
    get_info_path_gen()