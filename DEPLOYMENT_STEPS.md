# Step-by-Step Deployment Guide for Bandit Test API

This guide will walk you through deploying your Multi-Armed Bandit Test Optimization System to Heroku.

## Prerequisites

1. Create a [Heroku account](https://signup.heroku.com/) if you don't have one
2. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
3. Make sure you have Git installed

## Step 1: Prepare Your Application

Your application is already prepared for deployment with the following files:
- `Procfile`: Tells Heroku how to run your application
- `requirements.txt`: Lists all the Python dependencies
- `.env`: Contains environment variables (not to be committed to Git)

## Step 2: Login to Heroku

Open your terminal and run:

```bash
heroku login
```

Follow the prompts to log in to your Heroku account.

## Step 3: Create a Heroku Application

```bash
cd /Users/dmitryintoxic/Documents/Programs/bandit_test
heroku create your-app-name  # Choose a unique name
```

Replace `your-app-name` with a unique name for your application. This will be part of your application's URL: `https://your-app-name.herokuapp.com`.

## Step 4: Set Up a PostgreSQL Database

```bash
heroku addons:create heroku-postgresql:essential-0
```

This creates a PostgreSQL database for your application.

## Step 5: Configure Environment Variables

Set the necessary environment variables on Heroku:

```bash
heroku config:set FLASK_ENV=production
heroku config:set FLASK_DEBUG=False
heroku config:set ALLOWED_ORIGINS="*"  # Or your specific frontend domain
heroku config:set SECRET_KEY="your-secure-random-key"
```

Replace `your-secure-random-key` with a secure random string.

## Step 6: Deploy Your Application

If your project is not already in a Git repository, initialize one:

```bash
git init
git add .
git commit -m "Initial commit for deployment"
```

Then push to Heroku:

```bash
git push heroku main  # or git push heroku master
```

If you're using a different branch, you can push it to Heroku's main branch:

```bash
git push heroku your-branch-name:main
```

## Step 7: Initialize the Database

```bash
heroku run python init_db.py
```

This will create the necessary database tables and import initial questions.

## Step 8: Open Your Application

```bash
heroku open
```

Your API will now be accessible at `https://your-app-name.herokuapp.com/`

## Step 9: Update Your Frontend

Update your frontend application to use the new API URL:

1. For the Next.js frontend (career-test-nextjs), update the API_URL in `src/app/services/api.ts`:

```typescript
const API_URL = 'https://your-app-name.herokuapp.com';
```

2. Deploy your frontend to a platform like Vercel or Netlify.

## Troubleshooting

- If your application fails to start, check the logs: `heroku logs --tail`
- If database connections fail, verify your DATABASE_URL environment variable
- For CORS issues, ensure your ALLOWED_ORIGINS is set correctly

## Monitoring and Maintenance

- Check application logs: `heroku logs --tail`
- Monitor application performance in the Heroku Dashboard
- Set up alerts for application errors