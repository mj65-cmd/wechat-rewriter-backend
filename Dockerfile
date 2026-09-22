# Render 免费档 Docker 部署 (锁定 Python 3.11, 规避默认 3.7 装不动现代依赖的问题)
FROM python:3.11-slim

WORKDIR /app

# 先装依赖层 (利用 Docker 缓存)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再放源码与前端静态文件
COPY main.py .
COPY static ./static

EXPOSE 8000

# Render 会把 PORT 注入环境变量
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
