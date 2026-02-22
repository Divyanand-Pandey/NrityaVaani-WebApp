# 🚀 NatyaMudra Deployment Guide

To deploy NatyaMudra so that anyone in the world can access it, you will need to host the application. We have combined the **Backend (FastAPI)** and the **Frontend (React)** into a single, optimized Docker container. This is extremely cost-effective and easy to deploy on platforms like **Railway** or **Render**.

Because the backend requires large `.pth` model weights and OpenCV processes, we strongly recommend **Railway.app** as it provides excellent Docker support and sufficient RAM.

---

## Deploying to Railway (Recommended)

1. Ensure your codebase (including `backend/nrityavaani_mobilenet.pth` and `backend/hand_landmarker.task`) is pushed to a **GitHub Repository**.
2. Go to [Railway.app](https://railway.app) and create an account using GitHub.
3. Click **New Project** and select **Deploy from GitHub repo**.
4. Select your NatyaMudra repository.
5. Railway will automatically detect the `Dockerfile` at the root of your repository and begin building the unified full-stack application.
6. **Environment Variables**: Go to your new service -> Variables -> **New Variable**:
   - `GEMINI_API_KEY` : `(Paste your Gemini API key from api.bin here)`
7. **Generate a Domain**: Go to your service -> Settings -> Networking -> Click **Generate Domain**.
8. Wait for the build and deployment to finish (usually 3-5 minutes).
9. Your application is now live at the generated `.up.railway.app` URL!

---

## Deploying to Render (Alternative)

1. Go to [Render.com](https://render.com) and create an account using GitHub.
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.
4. Render should automatically detect your `Dockerfile`. If asked for environment, select **Docker**.
5. **Environment Variables**: Add an environment variable:
   - `GEMINI_API_KEY` : `(Paste your Gemini API key from api.bin here)`
6. Click **Deploy Web Service**. Render provides a `.onrender.com` URL when finished.

### 🎉 Done!
Your perfectly formatted AI Dance Application is now live on the internet! Users can click the link on their phones, tablets, or laptops to automatically use their webcam and hear the AI teacher guide them!
