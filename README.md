# T-bank logo detection

Решение задачи детекции логотипа Т-банка. Модель обучена с использованием YOLOv8, REST API реализован на FastAPI, упакован в Docker.  

## Инструкциями по сборке и запуску Docker контейнера
1. Клонировать репозиторий
   git clone https://github.com/juliashche/T-bank_logo_detection
   cd T-bank_logo_detection
3. Веса модели уже находятся в папке weights
4. Собрать и запустить контейнер
   docker build -t tbank-logo-api . 
   docker run -p 8000:8000 tbank-logo-api
5. После запуска сервис доступен на http://localhost:8000/docs
   
