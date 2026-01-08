import subprocess
import sys

def train_model():
    print("Trening modelu...")
    result = subprocess.run([sys.executable, "faces_train.py"])
    if result.returncode != 0:
        print("Błąd podczas treningu.")
        sys.exit(1)
    print("Trening zakończony.")

def run_main():
    print("Uruchamianie głównej aplikacji...")
    return subprocess.run([sys.executable, "Project_main_file.py"]).returncode

def main():
    while True:
        train_model()
        exit_code = run_main()

        if exit_code == 99:
            print("Wykryto nową osobę – ponowne trenowanie.")
            continue
        elif exit_code == 0:
            print("Program zakończony bez dodania nowej osoby.")
            break
        else:
            print(f"Nieoczekiwany kod wyjścia: {exit_code}")
            break

if __name__ == "__main__":
    main()
