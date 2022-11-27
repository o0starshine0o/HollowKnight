1. 检查代码
   1. 保证import都正常
2. 看作者介绍
   1. resnet blocks 是什么?
   2. 池化之后说是要加几层,为什么(性能上可能好点),什么效果
   3. 破防是什么情况
   4. 为什么要用2个大脑
      1. 看起来像是左右手操作,但是这样不利于协同操作
      2. 后期看下能不能像Dueling那样, 把两个网络整合起来
   5. 通过内存查找角色和BOSS状态?有木有更好的方法?
      1. 使用OpenCV和mod
   6. 多线程的运用
3. 运行HollowKnight
   1. lite HollowKnight 是什么
      1. switch版本的, 没什么用
   2. 存档位置
      1. /Users/huyongsheng/Library/Application\ Support/unity.Team\ Cherry.Hollow\ Knigh
      2. 带bak的是游戏里的自动存档，每次坐椅子或者触发其他存档事件（比如某些boss战，或者送花，等等）就会自动保存一个带编号的bak文件，游戏退出的时候会把最新的bak文件结合当前状态存入user*.dat（这是你每次进游戏读取的东西）。
      3. 进入寻神者,直接挑战BOSS
   3. 如何修改存档,让所有玩法都解锁
      1. 找到解锁了所有玩法的存档, 项目里就有, 找了半天..
      2. mod修改:https://www.wolai.com/2oCyqkQtMrrLNgqYsZaTdE
      3. mods的保存位置: /Users/huyongsheng/Library/Application\ Support/Steam/steamapps/common/Hollow\ Knight/hollow_knight.app/Contents/Resources/Data/Managed/Mods
      4. mod安装失败的log: /Users/huyongsheng/Library/Application Support/unity.Team Cherry.Hollow Knight/ModLog.txt
      5. 尝试一个mod: 伤害值显示, DamagedValue, 2个地方都需要文件才能正常运行
      6. 显示HP, MP等信息
         1. 显示更多信息, 比如角色状态, 精灵边界
         2. 如何把相应的信息读取到input中
         3. 如何做出自己的mod, 直接能够把相关信息输出
         4. 查看debug的mod
      7. 创建mod: https://space.bilibili.com/290906064/article
         1. 安装开发环境: 
         2. 按照教程, 制作一个空的mod
         3. 加载Debug的Mod
         4. 查看DebugMod是怎么拿到那些数据的
         5. 尝试输出csv文件
         6. DebugMod的源码: https://github.com/seanpr96/HollowKnight.Modding/blob/master/Assembly-CSharp/ModHooks.cs
         7. api文档: https://radiance.host/apidocs/Hooks.html
         8. 看看都能输出哪些特征
            1. 灵魂获取
         9. 回调太多了, 尽快选取需要的特征
            1. 对象边界, 可以查看DebugMod的源码, 把DebugMod的源码弄下来看看
               1. 对象判定: HitboxRender::TryAddHitboxes
            2. 最关键的信息: 自己的位置, 敌人的位置, 子弹的位置, 敌我的状态, 边界
               1. knight-> 黄色
               2. enemy -> 红色, 包括了所有带伤害的东西, 都认为是敌人
               3. attack -> 青色
               4. terrain -> 绿色, 地形
               5. trigger -> 蓝色
               6. breakable -> 粉色, 可以打坏的
               7. gate -> 深蓝
               8. other -> 橙色
            3. 使用的是Collider2D类型, 用于碰撞检测, 包括以下几种子类
               1. BoxCollider2D, 圆角矩形, 忽略了圆角, 用点集
               2. PolygonCollider2D, 多边形, 点集
               3. EdgeCollider2D, 椭圆形, 点集
               4. CircleCollider2D, 圆形, 单独绘制的
            4. 如何把状态作为入参
               1. BoxCollider2D, 用2个点的坐标
               2. PolygonCollider2D, 用矩形包裹, 也是2个点的坐标
               3. EdgeCollider2D, 用矩形包裹,也是2个点的坐标
               4. CircleCollider2D, 用矩形包裹, 还是2个点的坐标
               5. enemy, 需要确定名称, 使用OneHot, 这里会把维度提升得很高, 前期维度应该比较低
               6. knight就一个
               7. attack可能有多个, 最大也就3个吧(黑吼, 深渊尖叫, 黑砸, 黑暗降临), 武器用的也是attack
               8. enemy最大限制20个(前期应该是够了), 如果不够, 就随机在预留的地方挑选
               9. 边界(有可能会存在掉落的情况), 用最多10个矩形吧
               10. knight的状态, 感觉bool型数据可以作为oneHot输入
                   1. 状态: 空闲, 空降, 跑步
                   2. 速度: 水平, 垂直
                   3. HP: 9
                   4. MP: 198
                   5. isInvuln: 是否受伤
                   6. accept input: 是否接受输入, 黑砸的情况下无法接受输入
                   7. dash: 是否冲刺
                   8. jumping: 是否跳跃
                   9. superDash: 是否黑冲
                   10. failing: 是否下落
                   11. recoiling: 是否后坐力
                   12. wall lock: 不清楚和墙有啥关系
                   13. wall jumping: 不清楚和墙有啥关系
                   14. wall touching: 是否到了墙边
                   15. wall sliding: 是否在墙上下滑
                   16. attacking: 是否正在攻击, 攻击状态下貌似不能按攻击键
                   17. canCast: 不清楚
                   18. canSuperDash: 是否可以黑冲
                   19. 就这些了,其他的也看不懂
            5. 如何编写Mod
               1. 代理ColliderCreateHook, 有新sprite时检测
               2. GUI里面的update每帧都会调用,看下调用时机
                  1. 如何保证自己的代码每帧都能运行到
               3. HitBox设置为一个单例, 场景改变时, 清空保存的Collider2D
                  1. 不能这么干, 会导致元素丢失的, 只能是删除对象, 再重建, 这是DebugMod的方案
                  2. 如何设置单例
                  3. 添加Collider2D
                  4. 删除Collider2D
                  5. 输出边界到log, 先输出knight的, 然后在考虑其他的情况
                     1. 输出的坐标不对, 可能是镜头camera的缘故, 看下DebugMod的源码, 矫正下位置
                  6. 输出BOSS的位置, 附带它的技能伤害位置
   4. 找到挑战的boss, 并且击败^_^
4. 设计思路
   1. 最小化验证模型
      1. 只要能分清楚敌我位置, 就把数据塞入DQN网络, 进行学习
         1. 输出knight的位置, √
         2. 输出boss的位置(可能有多个)
            1. 大黄蜂的位置, √
            2. 武器的位置(是个多边形), 需要使用map转为世界坐标才能知道具体该怎么用
               1. 多边形每次都是5个, 是不是有些已经不在画面内了啊, 输出knight的坐标参考下
               2. 先输出多边形, 再输出矩形, 看下效果对不对
            3. 技能的位置(是个圆形)
         3. 不要放到log里, 而是通过跨进程的方式, 考虑socket等方式
            1. pipe, 管道, 半双工, 单向流动, 父子进程, ×
            2. NamedPipe, 命名管道, 半双工, ×
            3. 信号量, 锁机制, 同步手段, ×
            4. 消息队列, 需要操作系统提供队列, 跨平台性能不够, ×
            5. 共享内存, 需要额外的控制, ×
            6. socket, 套接字, 考虑下, DomainSocket, 域套接字, 通过文件的方式进行, √
               1. 了解python的多进程
                  1. 实现多进程之间的通信, √
                  2. 建立一个服务端, 等待其他进程传送数据, √
               2. 额外学习了π的计算方法, 算是意外惊喜, √
               3. C# 创建一个客户端, 每一帧的log输出改成使用socket输出, √
                  1. 可以先使用简单的数据结构, 等能够跑通了, 再上定义的协议, √
                  2. UnixDomainSocketEndPoint在空洞骑士上无法使用, 看看有木有替代方法, 不行就只能用ip了, ×
                  3. UnixDomainSocketEndPoint是新版本的内容, DebugMod用的是老版本, 所以只能用ip和端口了, √
                  4. 能够连续输出数据, 目前卡在这了, 直接接收到一次数据, ×
                     1. 面向连接, 面向非连接, √
                     2. 有阻止, 无阻止, √
         4. 通过Python读取socket, 恢复数据
            1. 读取数据, √
            2. 恢复数据
         5. 通过Python向HollowNight输入动作(左, 右, jump)
            1. python怎么模拟键盘: https://www.ailibili.com/?p=560
            2. Mac会拦截, 需要授予权限, √
            3. pyautogui还有额外的功能: 屏幕快照, 可以考虑作为录制视频的工具, √
            4. pyautogui怎么支持第二块屏幕, √
               1. 使用工具后发现是x为负坐标, √
            5. pyautogui在mac环境下不能获取window, 那就需要固定HollowKnight的串口位置, ×
               1. HollowKnight只需要键盘操作就行了, √
            6. 尝试在游戏中完成移动, 跳跃的功能, √
         6. 能够完整的联调起来
            1. 当进入到大黄蜂场景时, 开始接受数据, 给出操作, √
            2. 当离开大黄蜂场景时, 执行脚本, 再次开始操作
            3. 如何跳出循环?
               1. 每局结束之后, 空出一段时间来结束python任务
         7. 使用DQN来训练
            1. 先来一个简单的DQN, √
            2. 在游戏输出和AI输入之间存在一个较大的时间延迟, 可能是print函数带来的, 关闭试试, ×
               1. 关闭print没有太大改善, ×
                  1. 无太大改善, 可以进行print, √
               2. 看下输入和返回的时间间隔多大(aps:action pre second), √
                  1. aps=1时, 0.5ms, √
                  2. dps=10, √
                     1. 发生了数据粘黏现象, ×
               3. 寻找其他解决方案, √
                  1. 使用send-receive机制防止粘包, √
                     1. 完成一次动作需要500ms, 时间有点长, × 
                     2. 貌似时sleep和计算时间的函数有问题, 经过调整, 平均耗时为0.5ms, 完全可以胜任, √
            3. 拆分输入
               1. 是否需要json来传递?, √
                  1. 方便添加数据, √
                  2. 但是效率比较慢, ×
                  3. json无法打包, ×
                     1. HollowKnight 使用的是Newtonsoft.Json, 那就尝试使用这个json工具吧, √
               2. python序列化json, √
               3. 拆分成不同的动作执行, √
               4. 运用模拟程序运行, √
               5. 使用游戏运行, √
         8. 使用DQN来完成躲避boss, 并且尝试进攻
            1. 把knight的坐标和enemies的坐标给dqn, √
               1. 需要先分解数据, 修整数据, √
               2. 
            2. 从dqn获取action, √
               1. shape报错了, 与规定的shape不一致, ×
                  1. 通过mod输出json,查看到底是为什么shape不对, √
                     1. 估计是还没有敌人, 只有knight, √
                        1. 强制设置shape, 让一个对象循环使用, 不要new新对象, √
            3. 设置reward
               1. knight的生命丢失, √
               2. boss的生命丢失, √
            4. 设置经验池
               1. 优先做, learn需要利用到经验池, √
                  1. 具体多大
                     1. 每秒60帧, 一局大约30秒, 2倍设置吧, √
                  2. 需要保存哪些数据
                     1. index, 索引, √
                     2. state, 当前游戏的状态, √
                     3. action, 选取的动作, √
                     4. reward, 得到的奖励, √
                        1. 需要C#把当前的生命值传入
                        2. 先简单些, 给一个假设的奖励, 后期再补, √
                        3. 奖励是上一步的奖励, 需要注意, √
                  3. 完成recall方法, √
            5. 完成learn函数, 让dqn开始学习
               1. 每次退出BOSS房间, 开始学习, √
               2. 尝试再次进入BOSS房间, 继续训练, √
               3. 完成一段时间的学习后使用脚本,再次进入BOSS房间, √
                  1. 启用一个新线程或者进程, 延迟一段时间后模拟按键, √
                  2. 提供一个退出的方案, 稍微挪动下, 不要出挑战面板就可以了, k代表跳跃, 小跳一下说明循环结束了, √
            6. 保存模型
               1. 经验池, √
               2. 当前的模型参数, √
            7. 读取模型
               1. 经验池, √
               2. 当前的模型参数, √
            8. 想办法检测学习效果
            9. 其他问题
5. 开始真正的AI
   1. 验证模型可学习
      1. 选择攻击键, 只要执行了攻击就给reward, 一段时间后只出现攻击, 这就验证了模型可学习
         1. 执行了过多的None操作和左移操作
            1. 增大经验池, √
            2. 如果数据不够, 就采取随机动作, √
         2. 学习只发生在战斗之后, 学习效率太慢了
            1. 循环多次进行学习, 充分利用下一句的等待时间, √
            2. 采用线程的方式, 避免学习太慢, 无法处理下一场数据, √
            3. 离开游戏后, 保存当前数据, 输出保存的pool大小, √
            4. 输出里有个warning, shape不对, 需要调整下, √
               1. batch_size不要固定就行了, √
         3. 让模型跑到36000条数据吧
   2. 验证模型可以完成击杀
   3. 优化模型