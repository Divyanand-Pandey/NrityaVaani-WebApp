# 🚀 NatyaMudra Deployment Guide

To deploy NatyaMudra so that anyone in the world can access it, you will need to host the **Backend** (FastAPI) and the **Frontend** (React) on accessible servers. This is called a "Full Stack" deployment. 

Because the backend requires large `.pth` model weights and OpenCV processes, we will deploy it to **Render** or **Railway**. The frontend is a lightweight React app, which we will deploy to **Vercel** or **Netlify**.

---

## Part 1: Deploying the Backend (API)
We recommend **Render.com** or **Railway.app** because they provide free/cheap environments with enough RAM to run PyTorch safely.

### 1. Preparation
1. Ensure your `requirements.txt` is strictly up to date.
2. In your `backend` folder, create a new file named `Procfile` (no extension) with exactly this line of code:
   ```text
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
3. Your codebase (including `backend/nrityavaani_mobilenet.pth` and `backend/hand_landmarker.task`) must be pushed to a **GitHub Repository**.

### 2. Hosting on Render
1. Go to [Render.com](https://render.com) and create an account using GitHub.
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.
4. Fill out the settings:
   - **Root Directory**: `backend` (Important! This tells Render not to build the React app).
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Environment Variables**: Scroll down to the Environment tab and click **Add Environment Variable**:
   - `GEMINI_API_KEY` : `(Paste your Gemini API key from api.bin here)`
     > *Note: By placing your key here, it stays hidden and secure in the cloud compared to pushing `api.bin` to Github!*
6. Click **Deploy Web Service** and wait around 5 minutes.
7. Render will output a live URL (e.g., `https://natyamudra-api.onrender.com`). Save this securely.

---

## Part 2: Deploying the Frontend (UI)
We strongly recommend **Vercel** because it automatically understands and natively routes `Vite + React` apps flawlessly. 

### 1. Preparation
1. Open `frontend/src/App.jsx`.
2. Find the constant `API_URL` at the top of the file:
   ```javascript
   // Change from localhost to your new live backend URL:
   const API_URL = 'https://natyamudra-api.onrender.com/predict';
   ```
3. Commit this change to your GitHub Repository.

### 2. Hosting on Vercel
1. Go to [Vercel.com](https://vercel.com) and create an account via GitHub.
2. Click **Add New Project** and select your NatyaMudra repository.
3. Vercel will ask "What is the Root Directory?". Click `Edit` and select `frontend`.
4. Vercel will automatically detect `Vite` and fill out the Build commands for you (`npm run build`).
5. Click **Deploy**.
6. Wait 1 minute. Vercel will generate a high-speed, live URL (e.g., `https://natyamudra.vercel.app`).

### 🎉 Done!
Your perfectly formatted AI Dance Application is now live on the internet! Users can click the Vercel link on their phones, tablets, or laptops to automatically use their webcam and hear the AI teacher guide them!
