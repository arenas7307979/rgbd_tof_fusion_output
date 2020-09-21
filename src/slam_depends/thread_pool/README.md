ThreadPool
==========
Commit: 9a42ec1329f259a5f4881a291db1dcb8f2ad9040
------------------------------------------------
Ref: https://github.com/progschj/ThreadPool
-------------------------------------------
A simple C++11 Thread Pool implementation.

Basic usage:
```c++
// create thread pool with 4 worker threads
ThreadPool pool(4);

// enqueue and store future
auto result = pool.enqueue([](int answer) { return answer; }, 42);

// get result from future
std::cout << result.get() << std::endl;

```
