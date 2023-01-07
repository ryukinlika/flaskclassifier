# Simple Flask Nematode Classifier WebApp
## Table of contents
1. [Introduction](#introduction)
2. [Modifying This Work](#improve)
3. [Heroku Deployment](#heroku)
4. [Other Resources](#ref)


## Introduction <a name="introduction"></a>
A simple deep learning nematode classifier Webapp developed using Flask web framework. 
- Model used: EfficientNetV2B0
- Trained on Dataset with 1016 files belonging to 11 classes

Model trained as part of a research: [Deep learning models for automatic identification of plant-parasitic nematode](https://doi.org/10.1016/j.aiia.2022.12.002)

For further implementation, please citate the [research paper](https://doi.org/10.1016/j.aiia.2022.12.002). 

| Genus                 | # of Samples |
|-----------------------|--------------|
| Genus Criconema       | 4            |
| Genus Criconemoides   | 103          |
| Genus Helicotylenchus | 135          |
| Genus Hemicycliophora | 6            |
| Genus Hirschmaniella  | 130          |
| Genus Hoplolaimus     | 151          |
| Genus Meloidogyne     | 211          |
| Genus Pratylenchus    | 116          |
| Genus Radopholus      | 31           |
| Genus Trichodorus     | 44           |
| Genus Xiphinema       | 85           |
|             **Total** | 1016         |

## Modifying This Work <a name="improve"></a>
Either fork the repository or download it to your local system.

Please citate the [research paper](https://doi.org/10.1016/j.aiia.2022.12.002) if you intend to use it in a publication. 

The `static/uploads` folder contains uploaded images, and can be deleted after inferences.

To run this webapp locally: 
1. Create Python Virtual Environment
```
# example - create virtual environment with foldername: venv, then activate the virtual environment
python -m venv venv
venv\Scripts\Activate
```
2. Install dependencies as required, refer to `requirements.txt`
```
pip install flask gunicorn bootstrap-flask tensorflow numpy pandas python-dotenv 
```
3. Run the flask webapp
```
flask run --no-debugger
```

Model is stored in the `model` folder.

You can change the model used in this webapp, but do note that model size must be below the max file size of GitHub (100MB). 
Change this line in  `main.py` to refer the new model file name 
```
model = keras.models.load_model("model/effv2b0.h5")
```


## Heroku Deployment <a name="heroku"></a>
### Preliminaries
You need a Heroku account with a valid payment method use the Free Tier of the service. 

(optional) Heroku CLI helps in versioning and deploying using CMD. 

Notable files: 
1. `requirements.txt`, list of required dependencies for Heroku Deployment. Update this file in case of dependencies changes. 
   1. To update it you can use `pip freeze` command to lists all installed dependencies on the current venv. E.g. `pip freeze > requirements.txt`
2. `Procfile`, web framework and entry point for Heroku Deployment


### Deployment
You can either connect your GitHub account to Heroku (recommended) or manually deploy your local repository using Heroku CLI  

To manually deploy your new app via Heroku CLI 
1. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Login to heroku and init a new git repository
```
heroku login
git init
heroku git:remote -a app-name
git add -A
git commit -m "First app"
git push heroku master
```
Refer to: [Deploying with Git | Heroku Dev Center](https://devcenter.heroku.com/articles/git)



## Other Resources <a name="ref"></a>
- [How to Deploy a Flask App on Heroku](https://dev.to/techparida/how-to-deploy-a-flask-app-on-heroku-heb) by Trilochan Parida
