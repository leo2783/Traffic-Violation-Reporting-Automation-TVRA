#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QRadioButton>
#include <QButtonGroup>
#include <QCheckBox>
#include <QTextEdit>
#include <QFileDialog>
#include <QDir>
#include <QFileInfo>
#include <QCryptographicHash>
#include <QSet>
#include <QStringList>
#include <QTabWidget>
#include <QLineEdit>
#include <QSpinBox>
#include <QTextStream>
#include <random>
#include <algorithm>
#include <vector>

class FileComparatorApp : public QWidget {
public:
    FileComparatorApp(QWidget *parent = nullptr) : QWidget(parent) {
        setWindowTitle("資料處理與比對工具");
        resize(750, 850);
        auto *mainLayout = new QVBoxLayout(this);
        auto *tabs = new QTabWidget(this);

        tabs->addTab(createCompareTab(), "📁 1. 檔案比對");
        tabs->addTab(createDatasetTab(), "📊 2. 切割 Dataset");
        tabs->addTab(createNegativeSampleTab(), "🖼️ 3. 負樣本抽取");
        tabs->addTab(createYoloTab(), "🎯 4. YOLO 清洗");

        mainLayout->addWidget(tabs);
    }

private:
    // ==========================================
    // 分頁 1: 檔案比對
    // ==========================================
    QStringList sourceDirs; QString targetDirCompare; QLabel *lblSource, *lblTargetCompare;
    QButtonGroup *modeGroup; QCheckBox *chkDelete, *chkDeleteSourceMissing, *chkCleanInvalidCompare; QTextEdit *logBoxCompare;

    QWidget* createCompareTab() {
        QWidget *tab = new QWidget(); auto *layout = new QVBoxLayout(tab);

        QLabel *descLabel = new QLabel("💡 說明：比對來源與目標資料夾。支援「同步清理無效標註」以優化資料品質。", tab);
        descLabel->setStyleSheet("color: #555555; font-style: italic; margin-bottom: 5px;");
        layout->addWidget(descLabel);

        auto *pathLayout = new QHBoxLayout(); auto *btnSource = new QPushButton("1. 新增來源資料夾", tab);
        lblSource = new QLabel("來源數量: 0", tab); auto *btnTarget = new QPushButton("2. 設定目標資料夾", tab);
        lblTargetCompare = new QLabel("尚未設定", tab);
        pathLayout->addWidget(btnSource); pathLayout->addWidget(lblSource);
        pathLayout->addWidget(btnTarget); pathLayout->addWidget(lblTargetCompare); layout->addLayout(pathLayout);

        auto *modeLayout = new QHBoxLayout(); modeGroup = new QButtonGroup(tab);
        auto *rbNameExt = new QRadioButton("完整檔名", tab); auto *rbNameOnly = new QRadioButton("主檔名", tab); auto *rbContent = new QRadioButton("檔案內容(MD5)", tab);
        rbNameExt->setChecked(true); modeGroup->addButton(rbNameExt, 0); modeGroup->addButton(rbNameOnly, 1); modeGroup->addButton(rbContent, 2);
        modeLayout->addWidget(rbNameExt); modeLayout->addWidget(rbNameOnly); modeLayout->addWidget(rbContent); layout->addLayout(modeLayout);

        chkDelete = new QCheckBox("將目標中相符的檔案直接刪除", tab); chkDelete->setStyleSheet("color: red;"); layout->addWidget(chkDelete);
        chkDeleteSourceMissing = new QCheckBox("刪除來源中「不存在於目標」的檔案", tab); chkDeleteSourceMissing->setStyleSheet("color: #E25A1C; font-weight: bold;"); layout->addWidget(chkDeleteSourceMissing);

        chkCleanInvalidCompare = new QCheckBox("同步清理目標資料夾內的「無效標註檔與圖片」", tab);
        chkCleanInvalidCompare->setStyleSheet("color: #0078D7;"); layout->addWidget(chkCleanInvalidCompare);

        auto *actionLayout = new QHBoxLayout(); auto *btnCompare = new QPushButton("3. 開始比對與清理", tab);
        btnCompare->setStyleSheet("background-color: green; color: white; font-weight: bold;"); auto *btnReset = new QPushButton("重設", tab);
        actionLayout->addWidget(btnCompare); actionLayout->addWidget(btnReset); layout->addLayout(actionLayout);
        logBoxCompare = new QTextEdit(tab); logBoxCompare->setReadOnly(true); layout->addWidget(logBoxCompare);

        connect(btnSource, &QPushButton::clicked, this, [this](){ QString dir = QFileDialog::getExistingDirectory(this, "選擇"); if (!dir.isEmpty()) { sourceDirs.append(dir); lblSource->setText("來源數量: " + QString::number(sourceDirs.size())); }});
        connect(btnTarget, &QPushButton::clicked, this, [this](){ QString dir = QFileDialog::getExistingDirectory(this, "選擇"); if (!dir.isEmpty()) { targetDirCompare = dir; lblTargetCompare->setText("目標: " + QFileInfo(dir).fileName()); }});
        connect(btnReset, &QPushButton::clicked, this, [this](){ sourceDirs.clear(); targetDirCompare.clear(); lblSource->setText("來源數量: 0"); lblTargetCompare->setText("尚未設定"); logBoxCompare->clear(); });
        connect(btnCompare, &QPushButton::clicked, this, &FileComparatorApp::runCompare);
        return tab;
    }

    void runCompare() {
        if (targetDirCompare.isEmpty()) return; logBoxCompare->clear(); int mode = modeGroup->checkedId();

        if (chkCleanInvalidCompare->isChecked()) {
            QDir tDir(targetDirCompare); int cleaned = 0;
            for (const QString &txtFile : tDir.entryList({"*.txt"}, QDir::Files)) {
                QString absPath = tDir.absoluteFilePath(txtFile); QString baseName = QFileInfo(absPath).completeBaseName(); QFile file(absPath);
                if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                    QString content = QTextStream(&file).readAll(); QStringList lines = content.split('\n', Qt::SkipEmptyParts);
                    bool isValid = false; for (const QString& line : lines) { if (line.trimmed().split(' ').size() >= 5) { isValid = true; break; } }
                    file.close();
                    if (!isValid) {
                        QFile::remove(absPath);
                        for (const QString &ext : {".jpg", ".png", ".jpeg"}) { if (tDir.exists(baseName + ext)) tDir.remove(baseName + ext); }
                        cleaned++;
                    }
                }
            }
            logBoxCompare->append(QString("🧹 [清理] 已移除 %1 組無效標註 (空檔/無座標)").arg(cleaned));
        }

        if (sourceDirs.isEmpty()) return;

        QSet<QString> targetKeys;
        for (const QFileInfo &info : QDir(targetDirCompare).entryInfoList(QDir::Files | QDir::NoDotAndDotDot)) {
            targetKeys.insert((mode == 0) ? info.fileName() : (mode == 1) ? info.completeBaseName() : getFileHash(info.absoluteFilePath()));
        }
        QSet<QString> sourceKeys; int sourceDeleted = 0;
        for (const QString &dirPath : sourceDirs) {
            for (const QFileInfo &info : QDir(dirPath).entryInfoList(QDir::Files | QDir::NoDotAndDotDot)) {
                QString key = (mode == 0) ? info.fileName() : (mode == 1) ? info.completeBaseName() : getFileHash(info.absoluteFilePath());
                sourceKeys.insert(key);
                if (!targetKeys.contains(key) && chkDeleteSourceMissing->isChecked()) { QFile::remove(info.absoluteFilePath()); sourceDeleted++; }
            }
        }
        int matched = 0, missing = 0, targetDeleted = 0;
        for (const QFileInfo &info : QDir(targetDirCompare).entryInfoList(QDir::Files | QDir::NoDotAndDotDot)) {
            QString key = (mode == 0) ? info.fileName() : (mode == 1) ? info.completeBaseName() : getFileHash(info.absoluteFilePath());
            if (sourceKeys.contains(key)) { if (chkDelete->isChecked()) { QFile::remove(info.absoluteFilePath()); targetDeleted++; } else { matched++; } } else { missing++; }
        }
        logBoxCompare->append(QString("\n📊 比對統計 -> 來源刪除: %1 | 目標刪除: %2 | 相符保留: %3 | 缺失: %4").arg(sourceDeleted).arg(targetDeleted).arg(matched).arg(missing));
    }

    QString getFileHash(const QString &filePath) {
        QFile file(filePath); if (file.open(QFile::ReadOnly)) { QCryptographicHash hash(QCryptographicHash::Md5); if (hash.addData(&file)) return hash.result().toHex(); } return "";
    }

    // ==========================================
    // 分頁 2: 切割 YOLO Dataset
    // ==========================================
    QString dsSourceDir, dsTargetDir; QLabel *lblDsSource, *lblDsTarget; QSpinBox *spinTrainRatio; QTextEdit *logBoxDataset;

    QWidget* createDatasetTab() {
        QWidget *tab = new QWidget(); auto *layout = new QVBoxLayout(tab);
        auto *pathLayout = new QVBoxLayout(); auto *btnSource = new QPushButton("1. 選擇來源資料夾", tab); lblDsSource = new QLabel("尚未設定", tab);
        auto *btnTarget = new QPushButton("2. 選擇輸出資料夾", tab); lblDsTarget = new QLabel("尚未設定", tab);
        pathLayout->addWidget(btnSource); pathLayout->addWidget(lblDsSource); pathLayout->addWidget(btnTarget); pathLayout->addWidget(lblDsTarget); layout->addLayout(pathLayout);
        auto *ratioLayout = new QHBoxLayout(); ratioLayout->addWidget(new QLabel("Train 比例 (%):", tab));
        spinTrainRatio = new QSpinBox(tab); spinTrainRatio->setRange(1, 99); spinTrainRatio->setValue(80); ratioLayout->addWidget(spinTrainRatio); ratioLayout->addStretch(); layout->addLayout(ratioLayout);
        auto *btnSplit = new QPushButton("🎲 洗牌並產生 Dataset", tab); btnSplit->setStyleSheet("background-color: #E25A1C; color: white; padding: 6px;"); layout->addWidget(btnSplit);
        logBoxDataset = new QTextEdit(tab); logBoxDataset->setReadOnly(true); layout->addWidget(logBoxDataset);

        connect(btnSource, &QPushButton::clicked, this, [this](){ QString dir = QFileDialog::getExistingDirectory(this, "選擇"); if (!dir.isEmpty()) { dsSourceDir = dir; lblDsSource->setText(dir); }});
        connect(btnTarget, &QPushButton::clicked, this, [this](){ QString dir = QFileDialog::getExistingDirectory(this, "選擇"); if (!dir.isEmpty()) { dsTargetDir = dir; lblDsTarget->setText(dir); }});
        connect(btnSplit, &QPushButton::clicked, this, &FileComparatorApp::runDatasetSplit); return tab;
    }

    void runDatasetSplit() {
        if (dsSourceDir.isEmpty() || dsTargetDir.isEmpty()) return;
        logBoxDataset->clear();

        QDir rootDir(dsSourceDir);
        // 1. 自動偵測來源結構
        bool isYoloFormat = rootDir.exists("images") && rootDir.exists("labels");
        QString imgRoot = isYoloFormat ? rootDir.absoluteFilePath("images") : dsSourceDir;
        QString lblRoot = isYoloFormat ? rootDir.absoluteFilePath("labels") : dsSourceDir;

        // 2. 搜尋有效配對 (Image + Txt)
        std::vector<QString> validBaseNames;
        QStringList imgExtensions = {".jpg", ".png", ".jpeg"};

        // 為了生成 yaml，順便記錄所有出現過的 Class ID
        QSet<int> uniqueClasses;

        for (const QString &txt : QDir(lblRoot).entryList({"*.txt"}, QDir::Files)) {
            QString baseName = QFileInfo(txt).completeBaseName();
            bool hasImg = false;
            QString foundImgPath;

            for (const QString &ext : imgExtensions) {
                if (QFile::exists(imgRoot + "/" + baseName + ext)) {
                    hasImg = true;
                    foundImgPath = baseName + ext;
                    break;
                }
            }

            if (hasImg) {
                validBaseNames.push_back(baseName);

                // 讀取標註檔抓取 Class ID 用於 dataset.yaml
                QFile f(lblRoot + "/" + txt);
                if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
                    QTextStream in(&f);
                    while (!in.atEnd()) {
                        QString line = in.readLine().trimmed();
                        if (!line.isEmpty()) {
                            QStringList parts = line.split(' ');
                            if (!parts.isEmpty()) uniqueClasses.insert(parts[0].toInt());
                        }
                    }
                    f.close();
                }
            }
        }

        if (validBaseNames.empty()) {
            logBoxDataset->append("❌ 未找到有效的圖片與標註配對，請檢查路徑。");
            return;
        }

        // 3. 洗牌並建立目標目錄
        std::mt19937 g(std::random_device{}());
        std::shuffle(validBaseNames.begin(), validBaseNames.end(), g);

        QString imgT = dsTargetDir + "/images/train", imgV = dsTargetDir + "/images/val";
        QString lblT = dsTargetDir + "/labels/train", lblV = dsTargetDir + "/labels/val";
        QDir().mkpath(imgT); QDir().mkpath(imgV); QDir().mkpath(lblT); QDir().mkpath(lblV);

        // 4. 開始移動/複製檔案
        int trainRatio = spinTrainRatio->value();
        int trainCount = (validBaseNames.size() * trainRatio) / 100;
        int actT = 0, actV = 0;

        for (size_t i = 0; i < validBaseNames.size(); ++i) {
            QString b = validBaseNames[i];
            bool isT = (i < static_cast<size_t>(trainCount));

            // 找出圖片副檔名
            QString imgFile;
            for (const QString &ext : imgExtensions) {
                if (QFile::exists(imgRoot + "/" + b + ext)) { imgFile = b + ext; break; }
            }

            // 複製檔案
            QFile::copy(imgRoot + "/" + imgFile, (isT ? imgT : imgV) + "/" + imgFile);
            QFile::copy(lblRoot + "/" + b + ".txt", (isT ? lblT : lblV) + "/" + b + ".txt");

            if (isT) actT++; else actV++;
            if (i % 10 == 0) QApplication::processEvents(); // 防止大型資料集導致 UI 卡死
        }

        // 5. 自動產生 dataset.yaml
        QFile yaml(dsTargetDir + "/dataset.yaml");
        if (yaml.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&yaml);
            out << "# Dataset auto-generated by FileComparatorTool\n";
            out << "path: " << dsTargetDir << "\n";
            out << "train: images/train\n";
            out << "val: images/val\n\n";
            out << "names:\n";

            QList<int> classList = uniqueClasses.values();
            std::sort(classList.begin(), classList.end());
            if (classList.isEmpty()) {
                out << "  0: class0\n";
            } else {
                for (int id : classList) {
                    out << "  " << id << ": class_" << id << "\n";
                }
            }
            yaml.close();
            logBoxDataset->append("📄 已自動建立 dataset.yaml");
        }

        logBoxDataset->append(QString("✅ 切割完成！\n總計: %1 組\n訓練集(Train): %2\n驗證集(Val): %3")
                                  .arg(validBaseNames.size()).arg(actT).arg(actV));
    }

    // ==========================================
    // 分頁 3: 負樣本抽取器
    // ==========================================
    QString nsSourceDir, nsTargetDir; QLabel *lblNsSource, *lblNsTarget; QCheckBox *chkGenEmptyTxt, *chkMoveFile; QTextEdit *logBoxNs;

    QWidget* createNegativeSampleTab() {
        QWidget *tab = new QWidget(); auto *layout = new QVBoxLayout(tab);
        QLabel *descLabel = new QLabel("💡 說明：提取「無標註檔」或「標註內容為空」的圖片作為負樣本。", tab);
        descLabel->setStyleSheet("color: #555555; font-style: italic; margin-bottom: 5px;");
        layout->addWidget(descLabel);

        auto *pathLayout = new QVBoxLayout(); auto *btnSource = new QPushButton("1. 選擇圖片庫來源", tab); lblNsSource = new QLabel("尚未設定", tab);
        auto *btnTarget = new QPushButton("2. 選擇負樣本輸出資料夾", tab); lblNsTarget = new QLabel("尚未設定", tab);
        pathLayout->addWidget(btnSource); pathLayout->addWidget(lblNsSource); pathLayout->addWidget(btnTarget); pathLayout->addWidget(lblNsTarget); layout->addLayout(pathLayout);

        chkGenEmptyTxt = new QCheckBox("自動生成空白 .txt (用於 YOLO 負樣本訓練)", tab);
        chkGenEmptyTxt->setChecked(true); layout->addWidget(chkGenEmptyTxt);
        chkMoveFile = new QCheckBox("移動檔案 (預設為複製)", tab);
        chkMoveFile->setStyleSheet("color: #8B0000;"); layout->addWidget(chkMoveFile);

        auto *btnExtract = new QPushButton("🔍 開始抽取", tab);
        btnExtract->setStyleSheet("background-color: #6C3483; color: white; font-weight: bold; padding: 6px;"); layout->addWidget(btnExtract);
        logBoxNs = new QTextEdit(tab); logBoxNs->setReadOnly(true); layout->addWidget(logBoxNs);

        connect(btnSource, &QPushButton::clicked, this, [this](){ QString dir = QFileDialog::getExistingDirectory(this, "選擇"); if (!dir.isEmpty()) { nsSourceDir = dir; lblNsSource->setText(dir); }});
        connect(btnTarget, &QPushButton::clicked, this, [this](){ QString dir = QFileDialog::getExistingDirectory(this, "選擇"); if (!dir.isEmpty()) { nsTargetDir = dir; lblNsTarget->setText(dir); }});
        connect(btnExtract, &QPushButton::clicked, this, &FileComparatorApp::runNegativeExtraction); return tab;
    }

    void runNegativeExtraction() {
        if (nsSourceDir.isEmpty() || nsTargetDir.isEmpty()) return;
        logBoxNs->clear();

        QDir rootDir(nsSourceDir);
        QDir targetRootDir(nsTargetDir);

        // 1. 自動適應 YOLO 目錄結構 (尋找 images 與 labels)
        bool isYoloFormat = rootDir.exists("images");
        QString imgRoot = isYoloFormat ? rootDir.absoluteFilePath("images") : nsSourceDir;
        QString lblRoot = isYoloFormat ? rootDir.absoluteFilePath("labels") : nsSourceDir;

        QString tgtImgRoot = isYoloFormat ? targetRootDir.absoluteFilePath("images") : nsTargetDir;
        QString tgtLblRoot = isYoloFormat ? targetRootDir.absoluteFilePath("labels") : nsTargetDir;

        QStringList subDirs = QDir(imgRoot).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        if (subDirs.isEmpty()) subDirs.append(""); // 支援單層無 train/val 的結構

        int extractedCount = 0;

        for (const QString &sub : subDirs) {
            QDir imgSubDir(sub.isEmpty() ? imgRoot : imgRoot + "/" + sub);
            QDir lblSubDir(sub.isEmpty() ? lblRoot : lblRoot + "/" + sub);

            QDir tgtImgSubDir(sub.isEmpty() ? tgtImgRoot : tgtImgRoot + "/" + sub);
            QDir tgtLblSubDir(sub.isEmpty() ? tgtLblRoot : tgtLblRoot + "/" + sub);

            // 若目標資料夾需要子目錄，則建立
            if (!sub.isEmpty()) {
                tgtImgSubDir.mkpath(".");
                tgtLblSubDir.mkpath(".");
            } else if (isYoloFormat) {
                targetRootDir.mkpath("images");
                targetRootDir.mkpath("labels");
            }

            for (const QString &imgFile : imgSubDir.entryList({"*.jpg", "*.png", "*.jpeg"}, QDir::Files)) {
                QString baseName = QFileInfo(imgFile).completeBaseName();
                QString txtName = baseName + ".txt";
                QString lblPath = lblSubDir.absoluteFilePath(txtName);

                bool isNegative = false;

                // 2. 更嚴謹的判定：找不到標註檔，或是標註檔內沒有有效的坐標 (>=5個參數)
                if (!QFile::exists(lblPath)) {
                    isNegative = true;
                } else {
                    QFile txtFile(lblPath);
                    if (txtFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
                        QStringList lines = QTextStream(&txtFile).readAll().split('\n', Qt::SkipEmptyParts);
                        bool hasValidBBox = false;
                        for(const QString& line : lines) {
                            if(line.trimmed().split(' ').size() >= 5) { hasValidBBox = true; break; }
                        }
                        if (!hasValidBBox) isNegative = true;
                        txtFile.close();
                    }
                }

                // 3. 執行抽取 (搬移或複製)
                if (isNegative) {
                    QString srcImgPath = imgSubDir.absoluteFilePath(imgFile);
                    QString tgtImgPath = tgtImgSubDir.absoluteFilePath(imgFile);

                    if (chkMoveFile->isChecked()) {
                        QFile::rename(srcImgPath, tgtImgPath);
                        // 【修復】如果原本有一個無效的 txt，移動時應將原檔刪除，避免遺留垃圾
                        if (QFile::exists(lblPath)) QFile::remove(lblPath);
                    } else {
                        QFile::copy(srcImgPath, tgtImgPath);
                    }

                    if (chkGenEmptyTxt->isChecked()) {
                        QFile emptyTxt(tgtLblSubDir.absoluteFilePath(txtName));
                        emptyTxt.open(QIODevice::WriteOnly);
                        emptyTxt.close();
                    }

                    QString logName = sub.isEmpty() ? imgFile : sub + "/" + imgFile;
                    logBoxNs->append("🌑 [抽取] " + logName);
                    extractedCount++;
                    QApplication::processEvents(); // 避免 UI 卡死
                }
            }
        }
        logBoxNs->append(QString("\n✅ 負樣本抽取完成 | 共抽出: %1 張").arg(extractedCount));
    }

    // ==========================================
    // 分頁 4: YOLO 清洗
    // ==========================================
    QString yoloDir; QLabel *lblYoloDir; QCheckBox *chkCleanEmpty, *chkChangeClass, *chkRename, *chkZeroPad;
    QLineEdit *editClassId, *editRenamePrefix; QTextEdit *logBoxYolo;

    QWidget* createYoloTab() {
        QWidget *tab = new QWidget(); auto *layout = new QVBoxLayout(tab);
        auto *dirLayout = new QHBoxLayout(); auto *btnYoloDir = new QPushButton("選擇原始 YOLO 資料夾", tab); lblYoloDir = new QLabel("尚未設定", tab);
        dirLayout->addWidget(btnYoloDir); dirLayout->addWidget(lblYoloDir); layout->addLayout(dirLayout);

        chkCleanEmpty = new QCheckBox("1. 刪除無座標標註檔 (同步刪除圖片)", tab); chkCleanEmpty->setChecked(true); layout->addWidget(chkCleanEmpty);
        auto *classLayout = new QHBoxLayout(); chkChangeClass = new QCheckBox("2. 更改類別 ID 為:", tab); editClassId = new QLineEdit("0", tab); editClassId->setMaximumWidth(50);
        classLayout->addWidget(chkChangeClass); classLayout->addWidget(editClassId); classLayout->addStretch(); layout->addLayout(classLayout);

        auto *renameLayout = new QHBoxLayout();
        chkRename = new QCheckBox("3. 批次改名:", tab);
        renameLayout->addWidget(chkRename);
        renameLayout->addWidget(new QLabel("前綴(留空純數字):", tab));
        editRenamePrefix = new QLineEdit("image_", tab); editRenamePrefix->setMaximumWidth(80);
        renameLayout->addWidget(editRenamePrefix);
        chkZeroPad = new QCheckBox("數字補零(如:00001)", tab); chkZeroPad->setChecked(true);
        renameLayout->addWidget(chkZeroPad);
        renameLayout->addStretch(); layout->addLayout(renameLayout);

        auto *btnProcess = new QPushButton("🚀 執行清洗與改名", tab);
        btnProcess->setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;"); layout->addWidget(btnProcess);
        logBoxYolo = new QTextEdit(tab); logBoxYolo->setReadOnly(true); layout->addWidget(logBoxYolo);

        connect(btnYoloDir, &QPushButton::clicked, this, [this](){ QString dir = QFileDialog::getExistingDirectory(this, "選擇資料夾"); if (!dir.isEmpty()) { yoloDir = dir; lblYoloDir->setText(dir); }});
        connect(btnProcess, &QPushButton::clicked, this, &FileComparatorApp::runYoloProcess); return tab;
    }

    void runYoloProcess() {
        if (yoloDir.isEmpty()) return; logBoxYolo->clear();

        QDir root(yoloDir);
        // 預設尋找 images 與 labels，若無則退回單層資料夾模式
        QString imgRoot = root.exists("images") ? root.absoluteFilePath("images") : yoloDir;
        QString lblRoot = root.exists("labels") ? root.absoluteFilePath("labels") : yoloDir;

        QStringList subDirs = QDir(imgRoot).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        if (subDirs.isEmpty()) subDirs.append(""); // 處理沒有 train/val 子資料夾的單層結構

        int del = 0, mod = 0, ren = 0;

        // --- 1. 清洗與修改類別 ---
        for (const QString &sub : subDirs) {
            QDir imgSubDir(sub.isEmpty() ? imgRoot : imgRoot + "/" + sub);
            QDir lblSubDir(sub.isEmpty() ? lblRoot : lblRoot + "/" + sub);

            for (const QString &txtFile : lblSubDir.entryList({"*.txt"}, QDir::Files)) {
                QString absPath = lblSubDir.absoluteFilePath(txtFile);
                QString baseName = QFileInfo(absPath).completeBaseName();
                QFile file(absPath);
                if (!file.open(QIODevice::ReadWrite | QIODevice::Text)) continue;

                QStringList lines = QTextStream(&file).readAll().split('\n', Qt::SkipEmptyParts);
                bool isEmpty = true;
                for(const QString& line : lines) { if(line.trimmed().split(' ').size() >= 5) { isEmpty = false; break; } }

                if (chkCleanEmpty->isChecked() && isEmpty) {
                    file.remove();
                    for (const QString &ext : {".jpg", ".png", ".jpeg"}) {
                        if (imgSubDir.exists(baseName + ext)) imgSubDir.remove(baseName + ext);
                    }
                    del++; continue;
                }

                if (chkChangeClass->isChecked() && !isEmpty) {
                    file.resize(0); QTextStream out(&file); QString newId = editClassId->text().trimmed();
                    for (QString line : lines) {
                        QStringList parts = line.trimmed().split(' ');
                        if (parts.size() >= 5) { parts[0] = newId; out << parts.join(' ') << "\n"; }
                    }
                    mod++;
                }
                file.close();
            }
        }

        // --- 2. 智慧銜接改名 (只改不符合規則的) ---
        if (chkRename->isChecked()) {
            QString prefix = editRenamePrefix->text();
            bool padZero = chkZeroPad->isChecked();
            // 建立正則表達式，例如: "^image_\d{5}$"
            QRegularExpression regex(QString("^%1\\d{%2}$").arg(QRegularExpression::escape(prefix)).arg(padZero ? "5" : "+"));

            int maxVal = 0;
            // 2-1. 找尋目前子資料夾中全域最大編號
            for (const QString &sub : subDirs) {
                QDir imgSubDir(sub.isEmpty() ? imgRoot : imgRoot + "/" + sub);
                for (const QString &f : imgSubDir.entryList({"*.jpg", "*.png", "*.jpeg"}, QDir::Files)) {
                    QString base = QFileInfo(f).completeBaseName();
                    if (regex.match(base).hasMatch()) maxVal = std::max(maxVal, base.mid(prefix.length()).toInt());
                }
            }

            int nextNum = maxVal > 0 ? maxVal + 1 : 1;

            // 2-2. 開始針對不合規則的檔案改名
            for (const QString &sub : subDirs) {
                QDir imgSubDir(sub.isEmpty() ? imgRoot : imgRoot + "/" + sub);
                QDir lblSubDir(sub.isEmpty() ? lblRoot : lblRoot + "/" + sub);

                for (const QString &imgFile : imgSubDir.entryList({"*.jpg", "*.png", "*.jpeg"}, QDir::Files)) {
                    QString oldBase = QFileInfo(imgFile).completeBaseName();
                    if (!regex.match(oldBase).hasMatch()) {
                        QString ext = QFileInfo(imgFile).suffix();
                        QString newBase = prefix + (padZero ? QString("%1").arg(nextNum, 5, 10, QChar('0')) : QString::number(nextNum));

                        // 改圖片
                        imgSubDir.rename(imgFile, newBase + "." + ext);
                        // 改對應標註
                        if (lblSubDir.exists(oldBase + ".txt")) {
                            lblSubDir.rename(oldBase + ".txt", newBase + ".txt");
                        }
                        nextNum++;
                        ren++;
                    }
                }
            }
        }

        logBoxYolo->append(QString("✅ 清洗與改名完成 | 清理: %1 組 | 類別修改: %2 組 | 智慧改名: %3 組").arg(del).arg(mod).arg(ren));
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    FileComparatorApp window;
    window.show();
    return app.exec();
}