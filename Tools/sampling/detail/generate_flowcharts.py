import graphviz

def create_zh_flowchart():
    dot = graphviz.Digraph(comment='Sampling Flowchart (ZH)', format='png')
    dot.attr(rankdir='TB', size='8,8')
    
    # 節點設定
    dot.node('A', '開始\n(輸入圖片路徑)', shape='box', style='rounded,filled', fillcolor='#dae8fc')
    dot.node('B', '是否啟用信心度採樣?\n(Confident Sample)', shape='diamond', style='filled', fillcolor='#fff2cc')
    dot.node('C', 'YOLO 模型預測 & 排序\n(正向/負向)', shape='box', style='rounded,filled', fillcolor='#f8cecc')
    dot.node('D', 'MobileNetV3 提取特徵向量\n(Embedding)', shape='box', style='rounded,filled', fillcolor='#d5e8d4')
    dot.node('E', '計算餘弦相似度矩陣\n(Cosine Similarity)', shape='box', style='rounded,filled', fillcolor='#d5e8d4')
    dot.node('F', '標註數量比對\n(Box Counts Match)', shape='box', style='rounded,filled', fillcolor='#e1d5e7')
    dot.node('G', '判定重複項\n(相似度 > 閾值 0.95)', shape='diamond', style='filled', fillcolor='#fff2cc')
    dot.node('H', '保留非重複圖片清單', shape='box', style='rounded,filled', fillcolor='#dae8fc')
    dot.node('I', '結束', shape='box', style='rounded,filled', fillcolor='#dae8fc')
    
    # 建立連接線
    dot.edge('A', 'B')
    dot.edge('B', 'C', label=' 是')
    dot.edge('B', 'D', label=' 否')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    dot.edge('F', 'G')
    dot.edge('G', 'H', label=' 否 (非重複)')
    dot.edge('G', 'I', label=' 是 (過濾)')
    dot.edge('H', 'I')
    
    # 渲染儲存
    dot.render('sampling/detail/sampling_flowchart_zh', view=False, cleanup=True)

def create_en_flowchart():
    dot = graphviz.Digraph(comment='Sampling Flowchart (EN)', format='png')
    dot.attr(rankdir='TB', size='8,8')
    
    # 節點設定
    dot.node('A', 'Start\n(Input Image Paths)', shape='box', style='rounded,filled', fillcolor='#dae8fc')
    dot.node('B', 'Enable Confident Sample?', shape='diamond', style='filled', fillcolor='#fff2cc')
    dot.node('C', 'YOLO Inference & Sort\n(Positive/Negative)', shape='box', style='rounded,filled', fillcolor='#f8cecc')
    dot.node('D', 'MobileNetV3 Feature Extraction\n(Embedding)', shape='box', style='rounded,filled', fillcolor='#d5e8d4')
    dot.node('E', 'Calculate Cosine Similarity Matrix', shape='box', style='rounded,filled', fillcolor='#d5e8d4')
    dot.node('F', 'Match Box Counts', shape='box', style='rounded,filled', fillcolor='#e1d5e7')
    dot.node('G', 'Detect Duplicates\n(Similarity > Threshold 0.95)', shape='diamond', style='filled', fillcolor='#fff2cc')
    dot.node('H', 'Keep Unique Images List', shape='box', style='rounded,filled', fillcolor='#dae8fc')
    dot.node('I', 'End', shape='box', style='rounded,filled', fillcolor='#dae8fc')
    
    # 建立連接線
    dot.edge('A', 'B')
    dot.edge('B', 'C', label=' Yes')
    dot.edge('B', 'D', label=' No')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    dot.edge('F', 'G')
    dot.edge('G', 'H', label=' No (Unique)')
    dot.edge('G', 'I', label=' Yes (Filtered)')
    dot.edge('H', 'I')
    
    # 渲染儲存
    dot.render('sampling/detail/sampling_flowchart_en', view=False, cleanup=True)

if __name__ == '__main__':
    create_zh_flowchart()
    create_en_flowchart()
