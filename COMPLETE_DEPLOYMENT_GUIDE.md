# Complete Deployment Guide for Bandit Test System

This guide will walk you through deploying both the backend API and frontend application to make your Multi-Armed Bandit Test Optimization System accessible on the internet.

## Part 1: Deploying the Backend API to Heroku

### Prerequisites

1. Create a [Heroku account](https://signup.heroku.com/) if you don't have one
2. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
3. Make sure you have Git installed

### Step 1: Prepare Your Backend Application

Your backend application is already prepared for deployment with the following files:
- `Procfile`: Tells Heroku how to run your application
- `requirements.txt`: Lists all the Python dependencies
- `.env`: Contains environment variables (not to be committed to Git)

### Step 2: Login to Heroku

Open your terminal and run:

```bash
heroku login
```

Follow the prompts to log in to your Heroku account.

### Step 3: Create a Heroku Application

```bash
cd /Users/dmitryintoxic/Documents/Programs/bandit_test
heroku create your-app-name  # Choose a unique name
```

Replace `your-app-name` with a unique name for your application. This will be part of your application's URL: `https://your-app-name.herokuapp.com`.

### Step 4: Set Up a PostgreSQL Database

```bash
heroku addons:create heroku-postgresql:essential-0
```

This creates a PostgreSQL database for your application.

### Step 5: Configure Environment Variables

Set the necessary environment variables on Heroku:

```bash
heroku config:set FLASK_ENV=production
heroku config:set FLASK_DEBUG=False
heroku config:set ALLOWED_ORIGINS="*"  # Or your specific frontend domain
heroku config:set SECRET_KEY="your-secure-random-key"
```

Replace `your-secure-random-key` with a secure random string.

### Step 6: Deploy Your Backend Application

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

### Step 7: Initialize the Database

```bash
heroku run python init_db.py
```

This will create the necessary database tables and import initial questions.

### Step 8: Verify Your Backend API

```bash
heroku open
```

Your API will now be accessible at `https://your-app-name.herokuapp.com/`

## Part 2: Deploying the Next.js Frontend to Vercel

### Prerequisites

1. Create a [Vercel account](https://vercel.com/signup) if you don't have one
2. Install the [Vercel CLI](https://vercel.com/download) (optional)

### Step 1: Prepare Your Frontend Application

Update your frontend application to use the new API URL:

1. Navigate to your Next.js project:

```bash
cd /Users/dmitryintoxic/Documents/Programs/career-test-nextjs
```

2. Update the API_URL in `src/app/services/api.ts`:

```typescript
const API_URL = 'https://your-app-name.herokuapp.com';
```

Replace `your-app-name` with the name of your Heroku app.

3. Commit your changes:

```bash
git add .
git commit -m "Update API URL for production"
```

### Step 2: Deploy to Vercel

#### Option 1: Deploy via Vercel Dashboard (Recommended for first-time deployment)

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your Git repository (GitHub, GitLab, or Bitbucket)
4. Configure your project settings:
   - Framework Preset: Next.js
   - Build Command: (leave as default)
   - Output Directory: (leave as default)
5. Click "Deploy"

#### Option 2: Deploy via Vercel CLI

```bash
vercel
```

Follow the prompts to configure your deployment.

### Step 3: Configure Environment Variables (if needed)

If your frontend uses environment variables, configure them in the Vercel Dashboard:

1. Go to your project in the Vercel Dashboard
2. Click on "Settings" > "Environment Variables"
3. Add any required environment variables

### Step 4: Verify Your Deployment

Once deployment is complete, Vercel will provide you with a URL for your application (e.g., `https://your-project.vercel.app`).

## Part 3: Connecting Frontend and Backend

### Step 1: Update CORS Settings on Backend

If you're experiencing CORS issues, update the ALLOWED_ORIGINS environment variable on Heroku to include your Vercel domain:

```bash
heroku config:set ALLOWED_ORIGINS="https://your-project.vercel.app"
```

For multiple origins, separate them with commas:

```bash
heroku config:set ALLOWED_ORIGINS="https://your-project.vercel.app,https://your-custom-domain.com"
```

### Step 2: Custom Domain Setup (Optional)

#### For Heroku Backend:

1. Purchase a domain from a domain registrar
2. In Heroku Dashboard, go to your app > Settings > Domains
3. Add your domain
4. Configure DNS settings at your domain registrar as instructed by Heroku

#### For Vercel Frontend:

1. In Vercel Dashboard, go to your project > Settings > Domains
2. Add your domain
3. Configure DNS settings at your domain registrar as instructed by Vercel

## Troubleshooting

### Backend Issues

- If your application fails to start, check the logs: `heroku logs --tail`
- If database connections fail, verify your DATABASE_URL environment variable
- For CORS issues, ensure your ALLOWED_ORIGINS is set correctly

### Frontend Issues

- Check build logs in Vercel Dashboard
- Verify API URL is correctly set
- Test API connectivity from browser console

## Monitoring and Maintenance

### Backend (Heroku)

- Check application logs: `heroku logs --tail`
- Monitor application performance in the Heroku Dashboard
- Set up alerts for application errors

### Frontend (Vercel)

- Monitor deployments in Vercel Dashboard
- Set up status alerts in Vercel
- Use Vercel Analytics to track performance

## Continuous Deployment

### Backend (Heroku)

To set up continuous deployment:

1. Connect your GitHub repository to Heroku in the Heroku Dashboard
2. Enable automatic deploys from your main branch

### Frontend (Vercel)

Vercel automatically sets up continuous deployment when you connect your Git repository.

## Scaling

### Backend (Heroku)

When your application needs more resources:

```bash
heroku ps:scale web=2  # Scale to 2 web dynos
```

For database scaling, upgrade your PostgreSQL plan in the Heroku Dashboard.

### Frontend (Vercel)

Vercel automatically scales your frontend application as needed.