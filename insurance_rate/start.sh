#! /bin/bash

# 修改 proxychains 配置文件
sed -i '/socks4/d' /etc/proxychains.conf
sed -i '/proxy_dns/d' /etc/proxychains.conf
printf "socks5  172.25.66.13 10086\n\
localnet 10.0.0.95\n\
localnet 10.0.0.96\n\
localnet 10.0.0.97\n\
localnet 10.0.0.9\n\
localnet 10.0.0.19\n\
localnet 172.18.20.149\n\
localnet 172.18.30.84\n\
localnet 172.20.2.253\n\
localnet 172.20.2.3\n\
localnet 172.18.30.86\n\
localnet 172.18.40.82\n\
localnet 172.22.0.10\n\
localnet 172.25.10.139\n\
localnet 172.25.10.142\n\
localnet 172.25.10.145\n\
localnet 172.25.10.131\n\
localnet 10.0.0.103\n\
localnet 20.62.58.5\n\
localnet 10.0.0.54\n\
localnet 20.192.158.81" >> /etc/proxychains.conf



# 生成按日期命名的日志文件名
LOG_FILE="$APP_LOG_PATH/insurance_rate-$(date +%F).log" 

# 检查日志文件是否存在，如果不存在则创建它
if [ ! -f "$LOG_FILE" ]; then
    touch "$LOG_FILE"
fi

tail -f $LOG_FILE &

cd /root/app/

if [ $DEPLOY_ENV = "qas" ];then
    \cp  .env_qas .env
elif [ $DEPLOY_ENV = "prd" ];then
    \cp .env_prd .env
elif [ $DEPLOY_ENV = "dev" ];then
    \cp .env_dev .env  
fi

# 使用 proxychains 并将 uvicorn 的输出重定向到按日期命名的日志文件
proxychains uvicorn main:app --host 0.0.0.0 --port 8080 >> $LOG_FILE 2>&1 
