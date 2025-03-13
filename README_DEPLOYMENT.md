# Deployment Guide for Bandit Test API

This guide will help you deploy your Multi-Armed Bandit Test Optimization System to the internet using Heroku, a cloud platform that makes it easy to deploy and scale web applications.

## Prerequisites

1. Create a [Heroku account](https://signup.heroku.com/) if you don't have one
2. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
3. Make sure you have Git installed

## Step 1: Prepare Your Application

Your application has already been prepared for deployment with the following files:

- `Procfile`: Tells Heroku how to run your application
- `requirements.txt`: Lists all the Python dependencies
- `.env`: Contains environment variables (not to be committed to Git)

## Step 2: Create a Git Repository

If your project is not already in a Git repository:

```bash
cd /Users/dmitryintoxic/Documents/Programs/ANTIKVAR/bandit_test
git init
git add .
git commit -m "Initial commit for deployment"
```

## Step 3: Create a Heroku Application

```bash
heroku login
heroku create your-app-name  # Choose a unique name
```

## Step 4: Set Up a PostgreSQL Database

```bash
heroku addons:create heroku-postgresql:essential-0
```

## Step 5: Configure Environment Variables

Set the necessary environment variables on Heroku:

```bash
heroku config:set FLASK_ENV=production
heroku config:set FLASK_DEBUG=False
heroku config:set ALLOWED_ORIGINS="*"  # Or your specific frontend domain
heroku config:set SECRET_KEY="your-secure-random-key"
```

## Step 6: Deploy Your Application

```bash
git push heroku main  # or git push heroku master
```

## Step 7: Initialize the Database

```bash
heroku run python init_db.py
```

## Step 8: Open Your Application

```bash
heroku open
```

Your API will now be accessible at `https://your-app-name.herokuapp.com/`

## Connecting Your Frontend

Update your frontend application to use the new API URL:

```javascript
// Change from
const API_URL = 'http://localhost:5000';

// To
const API_URL = 'https://your-app-name.herokuapp.com';
```

## Continuous Deployment

To set up continuous deployment:

1. Connect your GitHub repository to Heroku in the Heroku Dashboard
2. Enable automatic deploys from your main branch

## Alternative Deployment Options

### DigitalOcean App Platform

DigitalOcean App Platform is another good option for deploying Flask applications:

1. Create a DigitalOcean account
2. Create a new App from the App Platform section
3. Connect your GitHub repository
4. Configure as a Python app with the command: `gunicorn api:app`
5. Add environment variables similar to Heroku

### AWS Elastic Beanstalk

For more advanced deployments, AWS Elastic Beanstalk provides more control:

1. Install the AWS EB CLI
2. Initialize your EB environment: `eb init`
3. Create an environment: `eb create`
4. Deploy: `eb deploy`

## Monitoring and Maintenance

- Check application logs: `heroku logs --tail`
- Monitor application performance in the Heroku Dashboard
- Set up alerts for application errors

## Troubleshooting

- If your application fails to start, check the logs: `heroku logs --tail`
- If database connections fail, verify your DATABASE_URL environment variable
- For CORS issues, ensure your ALLOWED_ORIGINS is set correctly

## Scaling

When your application needs more resources:

```bash
heroku ps:scale web=2  # Scale to 2 web dynos
```

For database scaling, upgrade your PostgreSQL plan in the Heroku Dashboard.