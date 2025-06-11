import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 清理一下，确保环境干净
plt.close('all')

# -------------------------------------------------------------
# 核心测试代码
# -------------------------------------------------------------
try:
    # 直接指定我们已确认 Matplotlib 能找到的字体文件路径
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 

    print(f"正在尝试使用字体文件: {font_path}")

    # 从文件创建字体属性对象
    my_font = fm.FontProperties(fname=font_path)

    if my_font:
         print(f"✅ 成功加载字体: {my_font.get_name()}")
    else:
        raise ValueError("无法从路径加载字体属性")

    # --- 开始绘图 ---
    plt.figure()

    # 在标题、标签和文本中都使用这个字体
    plt.title('中文测试标题', fontproperties=my_font, fontsize=20)
    plt.xlabel('X轴：横坐标', fontproperties=my_font, fontsize=14)
    plt.ylabel('Y轴：纵坐标', fontproperties=my_font, fontsize=14)
    plt.text(0.5, 0.5, '中文字符一切正常！', fontproperties=my_font, ha='center', fontsize=16)

    # 保存图片
    output_file = "chinese_test_plot.png"
    plt.savefig(output_file)
    print(f"\n🎉 绘图成功! 请立刻在文件浏览器中打开并检查图片: {output_file}")
    print("如果图片中的中文正常，我们就可以修复你的主代码了。")

except Exception as e:
    print(f"\n❌ 绘图失败: {e}")
    print("如果这一步失败，说明问题比预想的更复杂，可能是 Matplotlib 后端或库本身的问题。")
# -------------------------------------------------------------