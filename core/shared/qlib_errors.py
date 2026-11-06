class QlibRestartRequired(RuntimeError):
    def __init__(self, generation_id, current_generation_id):
        super().__init__("Qlib当前数据代与进程启动配置不一致，请重启后再读取或发布")
        self.generation_id = generation_id
        self.current_generation_id = current_generation_id
