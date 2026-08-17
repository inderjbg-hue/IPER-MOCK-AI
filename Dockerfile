FROM node:22-bookworm-slim

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY server ./server
COPY client ./client

ENV NODE_ENV=production
ENV PORT=8080
ENV DEV_MODE=false

EXPOSE 8080

CMD ["node", "server/server.js"]
