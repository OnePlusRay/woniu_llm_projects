import os
import pymysql
from typing import Dict, List, Any
from dotenv import load_dotenv

# 加载指定的环境变量文件
load_dotenv(".env_prd")

# 配置数据库连接
def get_db_connection():
    return pymysql.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USERNAME'),
        password=os.getenv('MYSQL_PASSWD'),
        database=os.getenv('MYSQL_DATABASE')
    )


def create_table_if_not_exists():
    '''检查并创建表格'''
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 创建表的SQL语句
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS error_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                request_id VARCHAR(255) NOT NULL,
                error_message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            # 执行创建表的SQL语句
            cursor.execute(create_table_sql)
        conn.commit()
        print("Checked and created table 'error_logs' if it did not exist.")
    except Exception as e:
        print(f"Failed to create table: {e}")
    finally:
        conn.close()


def log_error_to_db(request_id: str, error_message: str):
    '''将错误信息存储到数据库中'''
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO error_logs (request_id, error_message) VALUES (%s, %s)"
            cursor.execute(sql, (request_id, error_message))
        conn.commit()
    except Exception as e:
        print(f"Failed to log error to database: {e}")
    finally:
        conn.close()


def get_error_logs(request_id: str = None) -> List[Dict[str, Any]]:
    '''查询错误日志'''
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            if request_id:
                sql = "SELECT id, request_id, error_message, created_at FROM error_logs WHERE request_id = %s"
                cursor.execute(sql, (request_id,))
            else:
                sql = "SELECT id, request_id, error_message, created_at FROM error_logs"
                cursor.execute(sql)
            result = cursor.fetchall()
            error_logs = [{"id": row[0], "request_id": row[1], "error_message": row[2], "created_at": row[3]} for row in result]
            return error_logs
    except Exception as e:
        print(f"Failed to retrieve error logs: {e}")
        return []
    finally:
        conn.close()

if __name__ == "__main__":
    print(get_error_logs('2024'))