# create_project.py
import os
import sys
from pathlib import Path

def create_project_structure():
    """创建项目目录和文件结构"""
    
    project_name = "sleep_analysis"
    base_dir = Path(project_name)
    
    # 创建目录结构
    directories = [
        base_dir,
        base_dir / "data",
        base_dir / "analysis",
        base_dir / "visualization",
        base_dir / "recommendations",
        base_dir / "tests",
        base_dir / "scripts",
        base_dir / "docs",
    ]
    
    # 创建文件结构
    files = {
        base_dir / "config.py": config_py_content,
        base_dir / "main.py": main_py_content,
        base_dir / "requirements.txt": requirements_txt_content,
        base_dir / "setup.py": setup_py_content,
        base_dir / "README.md": readme_md_content,
        
        # data
        base_dir / "data" / "__init__.py": "",
        base_dir / "data" / "generate_data.py": data_generate_data_py_content,
        
        # analysis
        base_dir / "analysis" / "__init__.py": "",
        base_dir / "analysis" / "exploratory_analysis.py": analysis_exploratory_analysis_py_content,
        base_dir / "analysis" / "correlations.py": analysis_correlations_py_content,
        
        # visualization
        base_dir / "visualization" / "__init__.py": "",
        base_dir / "visualization" / "time_series.py": visualization_time_series_py_content,
        base_dir / "visualization" / "distributions.py": visualization_distributions_py_content,
        base_dir / "visualization" / "demographics.py": visualization_demographics_py_content,
        
        # recommendations
        base_dir / "recommendations" / "__init__.py": "",
        base_dir / "recommendations" / "sleep_advice.py": recommendations_sleep_advice_py_content,
        
        # tests
        base_dir / "tests" / "__init__.py": "",
        base_dir / "tests" / "test_data.py": tests_test_data_py_content,
        base_dir / "tests" / "test_analysis.py": tests_test_analysis_py_content,
        
        # scripts
        base_dir / "scripts" / "run_analysis.py": scripts_run_analysis_py_content,
        
        # docs
        base_dir / "docs" / "index.md": docs_index_md_content,
    }
    
    # 创建目录
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    # 创建文件并写入内容
    for file_path, content in files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"创建文件: {file_path}")
    
    print(f"\n项目 {project_name} 创建完成！")
    print(f"请进入目录: cd {project_name}")
    print(f"安装依赖: pip install -r requirements.txt")
    print(f"运行分析: python main.py")
    print(f"或使用: sleep_analysis (如果已通过setup.py安装)")

# 文件内容定义
config_py_content = '''# config.py
# 项目配置参数

class Config:
    # 数据生成参数
    NUM_USERS = 100
    NUM_DAYS = 30
    
    # 可视化参数
    PLOT_STYLE = "seaborn-whitegrid"
    COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
    PRIMARY_COLOR = "#2980b9"
    
    # 分析参数
    QUALITY_BINS = [0, 4, 6, 8, 10]  # 睡眠质量分箱
    SLEEP_TIME_BINS = [0, 20, 22, 24, 100]  # 睡眠时段分箱
    
    # 文件路径
    DATA_FILE = "data/sleep_data.csv"
    OUTPUT_DIR = "output/"
'''

main_py_content = '''# main.py
# 睡眠质量分析系统主程序

import pandas as pd
import os
from data.generate_data import generate_sleep_data
from analysis.exploratory_analysis import SleepExplorer
from analysis.correlations import SleepCorrelations
from visualization.time_series import plot_sleep_quality_over_time, plot_daily_patterns
from visualization.distributions import plot_sleep_distributions
from visualization.demographics import plot_demographic_analysis
from recommendations.sleep_advice import SleepAdvisor
from config import Config

def create_output_directory():
    """创建输出目录"""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

def main():
    print("="*50)
    print("睡眠质量分析与改善系统")
    print("="*50)
    
    # 1. 生成或加载数据
    data_file = Config.DATA_FILE
    if os.path.exists(data_file):
        print(f"\\n[步骤1/5] 加载现有数据 {data_file}...")
        sleep_df = pd.read_csv(data_file)
    else:
        print(f"\\n[步骤1/5] 生成模拟睡眠数据...")
        sleep_df = generate_sleep_data()
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        sleep_df.to_csv(data_file, index=False)
    print(f"数据加载完成! 数据集大小: {sleep_df.shape}")
    
    # 2. 探索性分析
    print("\\n[步骤2/5] 进行探索性分析...")
    explorer = SleepExplorer(sleep_df)
    explorer.process_data()
    describe_stats = explorer.describe_dataset()
    analysis_results = explorer.basic_stats()
    
    # 3. 相关性分析
    print("\\n[步骤3/5] 进行相关性分析...")
    correlator = SleepCorrelations(sleep_df)
    corr_matrix = correlator.correlation_matrix()
    correlation_results = correlator.analyze_factors()
    group_results = correlator.group_analysis()
    
    # 4. 创建输出目录
    create_output_directory()
    
    # 5. 可视化
    print("\\n[步骤4/5] 生成可视化图表...")
    plots = [
        ("sleep_quality_over_time.png", plot_sleep_quality_over_time(sleep_df)),
        ("daily_sleep_patterns.png", plot_daily_patterns(sleep_df)),
        ("sleep_distributions.png", plot_sleep_distributions(sleep_df)),
        ("demographic_analysis.png", plot_demographic_analysis(sleep_df))
    ]
    
    for filename, plot in plots:
        plot.savefig(os.path.join(Config.OUTPUT_DIR, filename))
        print(f"保存图表: {filename}")
    
    # 6. 生成建议
    print("\\n[步骤5/5] 生成睡眠改善建议...")
    advisor = SleepAdvisor(sleep_df, analysis_results, correlation_results)
    
    print("\\n" + "="*50)
    print("一般改善建议:")
    print(advisor.general_recommendations())
    
    print("\\n" + "="*50)
    print("基于数据分析的改善建议:")
    print(advisor.data_driven_recommendations())
    
    print("\\n" + "="*50)
    print("个性化建议 (用户 #42):")
    print(advisor.personalized_recommendations(42))
    
    print("\\n" + "="*50)
    print(f"分析完成! 可视化图表已保存到: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
'''

requirements_txt_content = '''# requirements.txt
numpy==1.26.0
pandas==2.1.1
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.2
scikit-learn==1.3.1
faker==19.3.0
'''

setup_py_content = '''# setup.py
from setuptools import setup, find_packages

setup(
    name='sleep_analysis',
    version='0.1.0',
    description='睡眠质量分析与改善系统',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.0',
        'pandas>=2.1.1',
        'matplotlib>=3.7.2',
        'seaborn>=0.12.2',
        'scipy>=1.11.2',
        'scikit-learn>=1.3.1',
        'faker>=19.3.0',
    ],
    extras_require={
        'dev': [
            'pytest',
            'coverage',
            'flake8',
            'black'
        ]
    },
    entry_points={
        'console_scripts': [
            'sleep_analysis=sleep_analysis.main:main'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False
)
'''

readme_md_content = ''# 睡眠质量分析与改善系统

create_project_structure()