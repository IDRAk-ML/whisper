### **Fixing cuDNN Not Found Error on Ubuntu 24.04 with CUDA 12.6**

If you're encountering errors related to missing `libcudnn_ops.so` while running a deep learning model, follow these steps to properly install and configure **cuDNN**.

---

## **Step 1: Download and Install cuDNN**
You have already downloaded cuDNN using:

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.7.0/local_installers/cudnn-local-repo-debian12-9.7.0_1.0-1_amd64.deb
```

Now, install it:

```bash
sudo dpkg -i cudnn-local-repo-debian12-9.7.0_1.0-1_amd64.deb
```

Add the repository key:

```bash
sudo cp /var/cuda-repo-debian12-9-7-local/cudnn-*-keyring.gpg /usr/share/keyrings/
```

Update package lists:

```bash
sudo apt-get update
```

Install cuDNN:

```bash
sudo apt-get -y install cudnn
```

---

## **Step 2: Verify cuDNN Installation**
After installation, check if the cuDNN libraries exist:

```bash
ls -l /usr/lib/x86_64-linux-gnu | grep libcudnn
ls -l /usr/local/cuda/lib64 | grep libcudnn
```

Expected output should include:

```
libcudnn_ops.so -> libcudnn_ops.so.9
libcudnn_ops.so.9 -> libcudnn_ops.so.9.7.0
libcudnn_ops.so.9.7.0
```

If the files are missing, reinstall cuDNN.

---

## **Step 3: Set Environment Variables**
Ensure that the cuDNN library path is included in `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

To confirm the changes:

```bash
echo $LD_LIBRARY_PATH
```

---

## **Step 4: Verify cuDNN and CUDA Compatibility**
Check your cuDNN version:

```bash
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

Check CUDA version:

```bash
nvcc --version
```

Verify PyTorch detects CUDA:


---



---

## **Step 6: Reload Library Cache**
After setting up everything, reload the library cache:

```bash
sudo ldconfig
```

Now, retry your script:

```python
segments, info = model.transcribe("sample-0.mp3", beam_size=5)
```

---

### **Final Checks**
If the issue persists, run:

```bash
which cudnn
echo $CUDA_HOME
```


**Reboot the system and try running the script again.** 🚀