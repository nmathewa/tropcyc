# Tropical cyclone YT content and scripts

# Installation of manim


only linux based installation is discussed

1. cloning and installing from official git repo
2. simple way is creating an venv (virtual envio) in the same folder and using for manim


```
git clone https://github.com/3b1b/manim.git
python3 -m venv ./
source bin/activate
pip3 install -r requirement.txt
python3 -m pip install -r requirements.txt
```

3. additionally add the env **manim** to the bashrc



# Running Manim

## General command

```bash
python3 -m manim pythonFile.py className -args
```


## classNames

Manim programs have a certain structure. The Python program file requires you to make classes for all your series of animations. If you make more than a few classes, you have to run commands for every class you make. Seperate videos are made for every class

## Args

Args are a list of arguements that can be stated when running the program. The most important agruements and it's explanations are provided in the GO TO GUIDE. I recommend to look at it later, and start with the tutorial.

