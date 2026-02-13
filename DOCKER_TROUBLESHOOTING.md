# 🐳 Docker Troubleshooting Guide

## ❌ **Error: Docker Desktop Not Running**

### **Error Message:**
```
unable to get image: error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.51/images/...": 
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

### **✅ Solution:**

#### **Step 1: Start Docker Desktop**
1. Open **Docker Desktop** application
2. Wait for it to fully start (whale icon should be steady, not animating)
3. Check if it says "Docker Desktop is running" in the status

#### **Step 2: Verify Docker is Running**
```powershell
# Check Docker version
docker --version

# Check if Docker daemon is running
docker ps
```

If `docker ps` works without errors, Docker is running! ✅

#### **Step 3: Try Build Again**
```powershell
docker-compose up --build
```

---

## ⚠️ **Warning: Version Obsolete**

### **Warning Message:**
```
the attribute `version` is obsolete, it will be ignored
```

### **✅ Fixed:**
- Removed `version: '3.8'` from `docker-compose.yml`
- This is now the default in newer Docker Compose versions

---

## 🔧 **Other Common Issues**

### **Issue 1: Docker Desktop Not Installed**

**Solution:**
1. Download Docker Desktop: https://www.docker.com/products/docker-desktop
2. Install and restart computer
3. Start Docker Desktop

---

### **Issue 2: Docker Desktop Starting Slowly**

**Symptoms:**
- Docker Desktop takes long to start
- "Docker Desktop is starting..." message

**Solution:**
1. Wait for Docker Desktop to fully start (2-5 minutes on first run)
2. Check system tray for Docker icon
3. Make sure WSL 2 is enabled (if on Windows)

---

### **Issue 3: WSL 2 Not Enabled (Windows)**

**Error:**
```
WSL 2 installation is incomplete
```

**Solution:**
```powershell
# Enable WSL 2 (run as Administrator)
wsl --install

# Restart computer
# Then start Docker Desktop again
```

---

### **Issue 4: Port Already in Use**

**Error:**
```
Error: bind: address already in use
```

**Solution:**
```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

---

### **Issue 5: Out of Memory**

**Error:**
```
no space left on device
```

**Solution:**
1. Docker Desktop → Settings → Resources
2. Increase Memory allocation (recommended: 4GB+)
3. Increase Disk image size
4. Apply & Restart

---

## ✅ **Quick Checklist**

Before building, make sure:

- [ ] Docker Desktop is **installed**
- [ ] Docker Desktop is **running** (not just installed)
- [ ] Docker Desktop shows "Docker Desktop is running" status
- [ ] `docker --version` works
- [ ] `docker ps` works (no errors)
- [ ] Port 8000 is available
- [ ] Model file exists: `models/weights/best.pt`

---

## 🚀 **Step-by-Step: First Time Setup**

### **1. Install Docker Desktop**
- Download: https://www.docker.com/products/docker-desktop
- Install and restart computer

### **2. Start Docker Desktop**
- Open Docker Desktop application
- Wait for "Docker Desktop is running" message

### **3. Verify Installation**
```powershell
docker --version
docker ps
```

### **4. Build Your Image**
```powershell
cd C:\Users\Crich Joved\OneDrive\Desktop\ai-banana-earlystage
docker-compose up --build
```

---

## 📞 **Still Having Issues?**

### **Check Docker Desktop Logs:**
1. Docker Desktop → Troubleshoot → View logs
2. Look for error messages

### **Restart Docker Desktop:**
1. Right-click Docker icon in system tray
2. Quit Docker Desktop
3. Start Docker Desktop again
4. Wait for it to fully start

### **Reset Docker Desktop:**
1. Docker Desktop → Settings → Troubleshoot
2. Click "Reset to factory defaults"
3. Restart Docker Desktop

---

**Good luck! 🍀**

