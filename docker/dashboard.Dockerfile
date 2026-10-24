FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY docs/dashboard.html /usr/share/nginx/html/index.html
