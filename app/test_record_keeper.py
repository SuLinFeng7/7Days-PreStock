from utils.record_keeper import RecordKeeper

def main():
    # 创建RecordKeeper实例
    keeper = RecordKeeper()
    
    # 运行测试
    success = keeper.test_write()
    
    if success:
        print("\n测试成功！文件写入正常。")
    else:
        print("\n测试失败！请检查错误信息。")

if __name__ == "__main__":
    main() 