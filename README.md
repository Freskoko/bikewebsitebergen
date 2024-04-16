# Bergen City Bike website

## Where is the information from?

From bergen city bike's website [link-to-website](https://bergenbysykkel.no/apne-data)

## How to run:

### dev:

download csv files

python3

poetry

flask

### docker:

docker build . -t apitest

docker run -d -p 5000:5000 --name apitestrun apitest

## DEV TODO:

move to dash instead of flask in order to take inputs from user on plots

change size of stop-circles to match how many cycles actually go there

move from csv to database
- delete csv files
- un-gitignore data dir?