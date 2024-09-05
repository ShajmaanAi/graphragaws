import os
import subprocess
os.system("echo 'hello world'")

folder="/media/devuser/4cb8da84-6521-4e0b-9aeb-436f495aba56/DefectAnalyser/graphrag_book_Updated"
query = "what is the data about"
s = subprocess.run(["python", "-m","graphrag.query","--method","global",query], cwd=folder, check=True)
# s = subprocess.getstatusoutput(f'ps -ef | grep python3')

print(s)
