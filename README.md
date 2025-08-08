
2025年8月8日GPT-5发布了，同时Cursor支持大促销，GPT-5免费用（Lanch Week）。 于是就有了一个这个项目，用Rust重写C版本的minimap2. 

chat目录下是我的对话过程，全程vibe coding. 没有编程，全是观察结果不对，让它改，算是提现下人类的存在感。

原来是直接冲着minimap2 比对去的，基本上结果也对了。但是我突然想到一个问题，是不是Rust项目的index可以给minimap2用呢，经过一波操作，发现还真没问题。最开始速度比C版本慢了2倍多，被我PUA之后，速度就基本一样了，大概慢个10%吧。

```bash
# minimap2 rust版本构建索引
./target/release/mm2rs index -d test/index.mmi test/genome.fa

# minimap2 rust版本的索引
minimap2 -v0 test/index.mmi test/test.fa
# chr8    600     6       598     +       chr8    145138636       5999346 5999938 592     592     60      tp:A:Pcm:i:114 s1:i:592        s2:i:44 dv:f:0.0006     rl:i:0

# minimap2自己的索引
minimap2 -v0 test/index test/test.fa    
# chr8    600     6       598     +       chr8    145138636       5999346 5999938 592     592     60      tp:A:Pcm:i:114 s1:i:592        s2:i:44 dv:f:0.0006     rl:i:0
```

当然比对结果基本上一样，除了dv有点区别。

```bash
target/release/mm2rs align test/index.mmi test/test.fa

#chr8    600     6       598     +       chr8    145138636       5999346 5999938 592     592     60      tp:A:Pcm:i:114 s1:i:592        s2:i:0  dv:f:0.0000     rl:i:0
```

