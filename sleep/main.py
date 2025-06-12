import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from faker import Faker
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子确保结果可复现
np.random.seed(42)
random.seed(42)
fake = Faker('zh_CN')

# 1. 模拟生成睡眠数据
def generate_sleep_data(num_users=100, num_days=30):
    """模拟生成睡眠数据"""
    
    data = []
    start_date = datetime.now() - timedelta(days=num_days)
    
    for user_id in range(1, num_users + 1):
        # 生成用户基本信息
        age = np.random.randint(18, 65)
        gender = np.random.choice(['男', '女'], p=[0.48, 0.52])
        occupation = np.random.choice(['学生', '白领', '蓝领', '自由职业', '退休'], 
                                   p=[0.2, 0.4, 0.2, 0.15, 0.05])
        
        # 生成用户的睡眠模式参数
        user_sleep_mean = np.random.normal(7.5, 1.0)  # 平均睡眠时间
        user_sleep_variability = np.random.uniform(0.5, 2.0)  # 睡眠时间波动性
        user_stress_level = np.random.normal(5.0, 1.5)  # 基础压力水平
        user_caffeine_effect = np.random.uniform(0.5, 2.0)  # 咖啡因敏感度
        
        # 生成30天的睡眠记录
        for day in range(num_days):
            date = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")
            
            # 工作日和周末的睡眠模式不同
            is_weekend = datetime.strptime(date, "%Y-%m-%d").weekday() in [5, 6]
            
            # 基础睡眠时间 (小时)
            sleep_duration = np.random.normal(
                user_sleep_mean + (1.5 if is_weekend else 0), 
                user_sleep_variability
            )
            
            # 睡眠时长在3-12小时范围内
            sleep_duration = max(min(sleep_duration, 12), 3)
            
            # 压力水平 (1-10)
            stress = min(max(1, np.random.normal(user_stress_level, 1.5)), 10)
            
            # 生成影响因素
            caffeine_intake = np.random.poisson(1.2)  # 咖啡因摄入量 (标准杯)
            alcohol_intake = np.random.choice([0, 0, 0, 1, 1, 2])  # 酒精摄入量 (标准杯)
            exercise_duration = max(0, np.random.normal(0.5, 0.3))  # 运动时长 (小时)
            screen_time = np.random.normal(3.5, 1.5)  # 睡前屏幕使用时间 (小时)
            sleep_environment = np.random.choice(['安静', '一般', '嘈杂'], p=[0.6, 0.3, 0.1])  # 睡眠环境
            
            # 计算睡眠质量 (0-10分)
            # 基础分 + 各种影响因素
            sleep_quality = 6.0
            # 睡眠时长影响 (最优为7-9小时)
            if 7 <= sleep_duration <= 9:
                sleep_quality += 1.5
            elif sleep_duration < 5 or sleep_duration > 10:
                sleep_quality -= 2.0
            elif sleep_duration < 6 or sleep_duration > 9:
                sleep_quality -= 1.0
                
            # 压力影响
            sleep_quality -= stress * 0.2
            
            # 咖啡因影响
            sleep_quality -= caffeine_intake * user_caffeine_effect * 0.3
            
            # 酒精影响
            sleep_quality -= alcohol_intake * 0.4
            
            # 运动影响
            sleep_quality += min(exercise_duration, 2) * 0.5
            
            # 屏幕时间影响
            sleep_quality -= min(screen_time, 4) * 0.3
            
            # 睡眠环境影响
            if sleep_environment == '嘈杂':
                sleep_quality -= 1.2
            elif sleep_environment == '一般':
                sleep_quality -= 0.4
            
            # 添加随机波动
            sleep_quality += np.random.normal(0, 0.5)
            
            # 确保睡眠质量在1-10之间
            sleep_quality = max(min(sleep_quality, 10), 1)
            
            # 入睡时间 (基于睡眠时长和睡眠时间)
            sleep_time = "22:" + str(np.random.randint(0, 59)).zfill(2)
            wake_time = (datetime.strptime(sleep_time, "%H:%M") + 
                         timedelta(hours=sleep_duration + np.random.normal(0, 0.3))
                        ).strftime("%H:%M")
            
            # 记录数据
            data.append([
                user_id, date, age, gender, occupation, sleep_duration, sleep_quality,
                stress, caffeine_intake, alcohol_intake, exercise_duration, 
                screen_time, sleep_environment, sleep_time, wake_time
            ])
    
    # 创建DataFrame
    columns = [
        '用户ID', '日期', '年龄', '性别', '职业', '睡眠时长(小时)', '睡眠质量(1-10)', 
        '压力水平(1-10)', '咖啡因摄入量(杯)', '酒精摄入量(杯)', '运动时长(小时)', 
        '屏幕使用时间(小时)', '睡眠环境', '入睡时间', '醒来时间'
    ]
    
    return pd.DataFrame(data, columns=columns)

# 2. 数据分析与可视化
class SleepAnalyzer:
    """睡眠质量分析器"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.process_data()
        
    def process_data(self):
        """数据处理"""
        # 添加工作日/周末列
        self.df['日期'] = pd.to_datetime(self.df['日期'])
        self.df['星期几'] = self.df['日期'].dt.day_name()
        self.df['周末'] = self.df['星期几'].isin(['Saturday', 'Sunday'])
        
        # 添加小时变量
        self.df['入睡小时'] = self.df['入睡时间'].apply(lambda x: int(x.split(':')[0]))
        self.df['醒来小时'] = self.df['醒来时间'].apply(lambda x: int(x.split(':')[0]))
        
        # 睡眠时段类别
        self.df['睡眠时段'] = pd.cut(
            self.df['入睡小时'],
            bins=[0, 20, 22, 24, 100],
            labels=['非常早', '早', '正常', '晚'],
            right=False
        )
        
        # 睡眠质量类别
        self.df['睡眠质量类别'] = pd.cut(
            self.df['睡眠质量(1-10)'],
            bins=[0, 4, 6, 8, 10],
            labels=['很差', '较差', '较好', '很好'],
            include_lowest=True
        )
        
        # 对分类变量进行编码
        self.label_encoders = {}
        for col in ['睡眠环境', '职业', '睡眠时段']:
            le = LabelEncoder()
            self.df[col + '_编码'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
    
    def overall_analysis(self):
        """总体分析"""
        print("="*50)
        print("睡眠数据总体分析")
        print("="*50)
        
        # 基本统计
        print(f"数据集大小: {self.df.shape}")
        print(f"用户数量: {self.df['用户ID'].nunique()}")
        print(f"记录天数: {self.df['日期'].nunique()}")
        print("\n睡眠质量描述统计:")
        print(self.df['睡眠质量(1-10)'].describe())
        
        # 按睡眠质量类别计数
        print("\n睡眠质量分布:")
        print(self.df['睡眠质量类别'].value_counts(normalize=True).sort_index() * 100)
        
        # 可视化
        plt.figure(figsize=(15, 10))
        plt.suptitle('睡眠质量总体分析', fontsize=16)
        
        # 睡眠质量分布
        plt.subplot(2, 2, 1)
        sns.histplot(data=self.df, x='睡眠质量(1-10)', bins=10, kde=True)
        plt.title('睡眠质量分布')
        plt.axvline(self.df['睡眠质量(1-10)'].mean(), color='red', linestyle='--', 
                   label=f'均值: {self.df["睡眠质量(1-10)"].mean():.2f}')
        plt.legend()
        
        # 睡眠时长分布
        plt.subplot(2, 2, 2)
        sns.histplot(data=self.df, x='睡眠时长(小时)', bins=15, kde=True)
        plt.title('睡眠时长分布')
        plt.axvline(7, color='green', linestyle='--', label='推荐最小值')
        plt.axvline(9, color='green', linestyle='--', label='推荐最大值')
        plt.axvline(self.df['睡眠时长(小时)'].mean(), color='red', linestyle='--', 
                   label=f'均值: {self.df["睡眠时长(小时)"].mean():.2f}')
        plt.legend()
        
        # 睡眠时间分布
        plt.subplot(2, 2, 3)
        sleep_counts = self.df['睡眠时段'].value_counts()
        plt.pie(sleep_counts, labels=sleep_counts.index, autopct='%1.1f%%')
        plt.title('睡眠时间分布')
        
        # 睡眠质量随时间变化
        plt.subplot(2, 2, 4)
        weekly_quality = self.df.groupby('日期')['睡眠质量(1-10)'].mean().reset_index()
        sns.lineplot(data=weekly_quality, x='日期', y='睡眠质量(1-10)')
        plt.title('睡眠质量随时间变化')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def demographic_analysis(self):
        """人口统计特征分析"""
        print("="*50)
        print("人口统计特征分析")
        print("="*50)
        
        # 按性别分析
        gender_quality = self.df.groupby('性别')['睡眠质量(1-10)'].mean()
        print("\n按性别平均睡眠质量:")
        print(gender_quality)
        
        # 按年龄分析
        print("\n年龄与睡眠质量的相关性:")
        age_corr = pearsonr(self.df['年龄'], self.df['睡眠质量(1-10)'])[0]
        print(f"相关系数: {age_corr:.3f}")
        
        # 按职业分析
        occupation_quality = self.df.groupby('职业')['睡眠质量(1-10)'].mean().sort_values()
        print("\n按职业平均睡眠质量:")
        print(occupation_quality)
        
        # 可视化
        plt.figure(figsize=(15, 8))
        plt.suptitle('人口统计特征分析', fontsize=16)
        
        # 性别与睡眠质量
        plt.subplot(2, 2, 1)
        sns.boxplot(data=self.df, x='性别', y='睡眠质量(1-10)')
        plt.title('性别与睡眠质量')
        
        # 年龄与睡眠质量
        plt.subplot(2, 2, 2)
        sns.regplot(data=self.df, x='年龄', y='睡眠质量(1-10)', scatter_kws={'alpha':0.3})
        plt.title(f'年龄与睡眠质量 (r = {age_corr:.3f})')
        
        # 职业与睡眠质量
        plt.subplot(2, 2, 3)
        sns.barplot(data=self.df, x='睡眠质量(1-10)', y='职业', errorbar=None, orient='h')
        plt.title('职业与睡眠质量')
        
        # 按周末和工作日分析
        plt.subplot(2, 2, 4)
        weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_quality = self.df.groupby('星期几')['睡眠质量(1-10)'].mean().reindex(weekday_labels)
        weekday_quality.plot(kind='bar')
        plt.title('一周中各天睡眠质量')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def factor_analysis(self):
        """影响因素分析"""
        print("="*50)
        print("睡眠质量影响因素分析")
        print("="*50)
        
        # 计算相关系数
        corr_matrix = self.df[[
            '睡眠质量(1-10)', '压力水平(1-10)', '咖啡因摄入量(杯)', '酒精摄入量(杯)', 
            '运动时长(小时)', '屏幕使用时间(小时)', '睡眠时长(小时)'
        ]].corr()
        
        print("\n睡眠质量与各变量的相关系数矩阵:")
        print(corr_matrix['睡眠质量(1-10)'].sort_values(ascending=False))
        
        # 可视化
        plt.figure(figsize=(15, 12))
        plt.suptitle('睡眠质量影响因素分析', fontsize=16)
        
        # 相关系数热力图
        plt.subplot(2, 2, 1)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('变量间相关系数矩阵')
        
        # 压力与睡眠质量
        plt.subplot(2, 2, 2)
        sns.scatterplot(
            data=self.df, 
            x='压力水平(1-10)', 
            y='睡眠质量(1-10)', 
            alpha=0.4
        )
        plt.title('压力水平与睡眠质量')
        
        # 屏幕时间与睡眠质量
        plt.subplot(2, 2, 3)
        sns.boxplot(
            data=self.df,
            x=pd.cut(self.df['屏幕使用时间(小时)'], bins=[0, 1, 2, 3, 4, 24]),
            y='睡眠质量(1-10)'
        )
        plt.title('屏幕使用时间与睡眠质量')
        plt.xlabel('屏幕使用时间(小时)')
        
        # 环境与睡眠质量
        plt.subplot(2, 2, 4)
        sns.boxplot(data=self.df, x='睡眠环境', y='睡眠质量(1-10)', order=['安静', '一般', '嘈杂'])
        plt.title('睡眠环境与睡眠质量')
        
        plt.tight_layout()
        plt.show()
    
    def improvement_recommendations(self, user_id=None):
        """生成改善建议"""
        print("="*50)
        print("睡眠质量改善建议")
        print("="*50)
        
        # 整体改善建议
        print("\n整体改善建议:")
        print("1. 保持规律的睡眠时间表，即使在周末")
        print("2. 创建舒适的睡眠环境：凉爽、黑暗、安静")
        print("3. 避免在睡前摄入咖啡因和酒精")
        print("4. 将屏幕使用时间限制在睡前至少1小时")
        print("5. 通过冥想或阅读来管理压力水平")
        print("6. 白天定期进行体育锻炼，但避免在睡前剧烈运动")
        
        # 根据人口统计特征的建议
        if user_id is not None and user_id in self.df['用户ID'].values:
            user_data = self.df[self.df['用户ID'] == user_id].iloc[0]
            print(f"\n用户 #{user_id} 的个性化建议：")
            print(f"- 年龄: {user_data['年龄']}, 性别: {user_data['性别']}, 职业: {user_data['职业']}")
            
            # 基于年龄的建议
            if user_data['年龄'] > 50:
                print("- 随着年龄增长，可能需要调整睡眠习惯：")
                print("  → 考虑午休但不要超过30分钟")
                print("  → 确保白天有足够的自然光线照射")
            
            # 基于性别的建议
            if user_data['性别'] == '女':
                print("- 女性激素水平会影响睡眠，经期前后多加注意")
            
            # 基于职业的建议
            if user_data['职业'] in ['白领', '蓝领']:
                print("- 作为职场人士，管理工作压力对睡眠很重要：")
                print("  → 尝试工作与生活分离")
                print("  → 建立睡前放松仪式")
        
        # 睡眠时长建议
        print("\n睡眠时长建议:")
        avg_sleep = self.df['睡眠时长(小时)'].mean()
        if avg_sleep < 7:
            print(f"- 平均睡眠时长 ({avg_sleep:.1f}小时) 不足，建议增加到7-9小时")
        elif avg_sleep > 9:
            print(f"- 平均睡眠时长 ({avg_sleep:.1f}小时) 过长，可能影响睡眠质量")
        else:
            print(f"- 平均睡眠时长 ({avg_sleep:.1f}小时) 在推荐范围内，继续保持！")
        
        # 咖啡因摄入建议
        avg_caffeine = self.df['咖啡因摄入量(杯)'].mean()
        if avg_caffeine > 2:
            print(f"- 平均咖啡因摄入量 ({avg_caffeine:.1f}杯) 过高，建议减少到每天2杯以下")
        
        # 环境改善建议
        noisy_count = self.df[self.df['睡眠环境'] == '嘈杂'].shape[0]
        if noisy_count > self.df.shape[0] * 0.2:
            print(f"- 有{noisy_count}条记录在嘈杂环境中睡眠，建议使用白噪音机器或耳塞")

# 3. 主程序
if __name__ == "__main__":
    print("开始生成模拟睡眠数据...")
    sleep_df = generate_sleep_data(num_users=100, num_days=30)
    print(f"生成数据完成! 记录数: {sleep_df.shape[0]}")
    
    # 保存数据
    sleep_df.to_csv('sleep_data.csv', index=False)
    
    # 初始化分析器
    analyzer = SleepAnalyzer(sleep_df)
    
    # 执行分析
    analyzer.overall_analysis()
    analyzer.demographic_analysis()
    analyzer.factor_analysis()
    analyzer.improvement_recommendations(user_id=42)