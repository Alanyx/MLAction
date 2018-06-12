

##List中extend和append的区别：
            list.append(object) 向列表中添加一个对象object
            list.extend(sequence) 把一个序列seq的内容添加到列表中
            1、使用append的时候，是将new_media看作一个对象，整体打包添加到music_media对象中。
            2、使用extend的时候，是将new_media看作一个序列，将这个序列和music_media序列合并，并放在其后面。
            result = []
            result.extend([1,2,3])
            print(result)
            result.append([4,5,6])
            print(result)
            result.extend([7,8,9])
            print(result)
            结果：
            [1, 2, 3]
            [1, 2, 3, [4, 5, 6]]
            [1, 2, 3, [4, 5, 6], 7, 8, 9]
            
            
##Python中使用pickle持久化对象
    pickle.dump(obj, file[, protocol])
    这是将对象持久化的方法，参数的含义分别为：
    obj: 要持久化保存的对象；
    file: 一个拥有 write() 方法的对象，并且这个 write() 方法能接收一个字符串作为参数。这个对象可以是一个以写模式打开的文件对象或者一个 StringIO 对象，或者其他自定义的满足条件的对象。
    protocol: 这是一个可选的参数，默认为 0 ，如果设置为 1 或 True，则以高压缩的二进制格式保存持久化后的对象，否则以ASCII格式保存。
    
    对象被持久化后怎么还原呢？pickle 模块也提供了相应的方法，如下：
    pickle.load(file)