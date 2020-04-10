//计时
class Time_Interval
{
public:
    /*  开始
     */
    inline void start()
    {
        _t = clock();
    }
    /*  结束
     */
    inline float end()
    {
        return (clock() - _t) / (float)CLOCKS_PER_SEC;
    }
    /* 输出
	 */
    inline void prompt(const std::string &str,bool isreset = false)
    {
        std::cout << str.c_str() << end() << "s" << std::endl;
        if(isreset)
            start();
    }

protected:
    time_t _t;
};