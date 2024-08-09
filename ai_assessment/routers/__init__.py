from routers.potential_client_check_router import potential_client_check_process_func
from routers.annual_report_quality_router import annual_report_quality_process_func
from routers.call_record_summary_router import call_record_summary_process_func
from routers.call_record_quality_router import call_record_quality_process_func
from routers.initial_call_quality_router import initial_call_quality_process_func
from routers.routers_daily_service_summary_router import daily_potential_client_check_process_func
from routers.routers_month_service_summary_router import month_potential_client_check_process_func

# 对应job_request 的参数：tag: str          # 会通过main程序中的process_func = tag_2_func_mapping[job_request.tag]中传入tag，再匹配跳转下面的五个函数中，都存在routers里
tag_2_func_mapping = {
    "ANNUAL_REPORT_QUALITY": annual_report_quality_process_func,        # 返回一个json_message，根据tag不同，可能会用到不同的模板导出不同的message
    "POTENTIAL_CHECK": potential_client_check_process_func,
    "SUMMARY": call_record_summary_process_func,
    "INITIAL_CALL": initial_call_quality_process_func,
    "POLICY_ANNUAL": call_record_quality_process_func,
    "DAILY_SERVICE_SUMMARY": daily_potential_client_check_process_func,
    "MONTH_SERVICE_SUMMARY": month_potential_client_check_process_func
}