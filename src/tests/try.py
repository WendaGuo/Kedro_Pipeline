import os

if __name__ == "__main__":
    cwd = os.getcwd()
    ind=cwd.find("src")
    print(cwd[:ind])