"""dronesift GUI — drop a folder of drone footage, analyse in background, review shots, export.

Run with:  uv sync --extra gui && uv run dronesift-gui    (or: uv run python -m dronesift.gui)

The analysis itself runs in QProcess workers driving the dronesift CLI, so a crash can never
take the GUI down and progress behaves exactly like the command line. Reviewing embeds the
generated viz.html (QtWebEngine when available, otherwise the system browser); shots are
toggled in a side panel and re-exported in-process with the lossless cutter.
"""
import json
import os
import sys
import time

try:
    from PySide6.QtCore import Qt, QProcess, QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                                   QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                                   QListWidgetItem, QMainWindow, QPlainTextEdit, QPushButton,
                                   QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
    HAVE_GUI = True
except ImportError:
    HAVE_GUI = False

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView        # optional: in-app review page
    from PySide6.QtCore import QCoreApplication
    HAVE_WEBENGINE = True
except ImportError:
    HAVE_WEBENGINE = False

WORKERS = 2
EXTS = ('.mp4', '.mov')

def scan_videos(folder):
    out = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(EXTS) and not name.startswith('.'):
            out.append(os.path.join(folder, name))
    return out

def fmt_dur(sec):
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"

class ReviewWindow(QMainWindow):
    """One flight: embedded viz.html on the left, shot checklist + re-export on the right."""

    def __init__(self, video, out_dir, shots, segs, fps, parent=None):
        super().__init__(parent)
        self.video, self.out_dir, self.fps = video, out_dir, fps
        self.shots = shots or segs
        base = os.path.splitext(os.path.basename(video))[0]
        self.setWindowTitle(f"审查 — {base}")
        split = QSplitter()
        html = os.path.join(out_dir, base + '_viz.html')
        if HAVE_WEBENGINE and os.path.exists(html):
            self.view = QWebEngineView()
            self.view.load(QUrl.fromLocalFile(html))
        else:
            self.view = QLabel('（未安装 QtWebEngine，点击下方按钮在浏览器中打开审查页面）')
            self.view.setWordWrap(True)
            if os.path.exists(html):
                QDesktopServices.openUrl(QUrl.fromLocalFile(html))
        split.addWidget(self.view)
        side = QWidget(); box = QVBoxLayout(side)
        box.addWidget(QLabel(f"{len(self.shots)} 个镜头（勾选 = 导出）"))
        self.list = QListWidget()
        for i, s in enumerate(self.shots):
            it = QListWidgetItem(f"#{i + 1:03d}  {fmt_dur(s['start'])}  {s['duration']:4.1f}s  {s.get('label', '')}")
            it.setCheckState(Qt.Checked)
            it.setData(Qt.UserRole, s)
            self.list.addItem(it)
        box.addWidget(self.list, 1)
        btns = QHBoxLayout()
        b_all = QPushButton('全选'); b_none = QPushButton('全不选')
        b_all.clicked.connect(lambda: self.set_all(Qt.Checked)); b_none.clicked.connect(lambda: self.set_all(Qt.Unchecked))
        btns.addWidget(b_all); btns.addWidget(b_none); box.addLayout(btns)
        b_exp = QPushButton('重新导出勾选镜头（无损 copy）'); b_exp.clicked.connect(self.reexport); box.addWidget(b_exp)
        b_dir = QPushButton('打开输出目录'); b_dir.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))); box.addWidget(b_dir)
        split.addWidget(side)
        split.setSizes([900, 340])
        self.setCentralWidget(split)
        self.resize(1400, 860)

    def set_all(self, state):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(state)

    def reexport(self):
        from .. import cut as cutmod, fcpxml
        sel = [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
               if self.list.item(i).checkState() == Qt.Checked]
        if not sel:
            return
        exp = os.path.join(self.out_dir, 'export'); os.makedirs(exp, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.video))[0]
        cutmod.write_edl(sel, self.video, os.path.join(exp, base + '_export.edl'), self.fps, digits=3)
        fcpxml.write(sel, self.video, os.path.join(exp, base + '_export.fcpxml'), self.fps)
        paths = cutmod.cut(self.video, sel, exp, 'copy', self.fps, digits=3)
        QDesktopServices.openUrl(QUrl.fromLocalFile(exp))
        return paths

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('dronesift — 航拍素材自动筛片')
        self.resize(1100, 700)
        self.queue = []            # dicts: video, out, row
        self.running = {}          # QProcess -> queue item
        self.done = []             # finished queue items (for review)
        central = QWidget(); root = QVBoxLayout(central)

        top = QHBoxLayout()
        b_add = QPushButton('添加文件夹…'); b_add.clicked.connect(self.add_folder); top.addWidget(b_add)
        top.addWidget(QLabel('输出后缀'))
        self.suffix = QLineEdit('_gui'); self.suffix.setMaximumWidth(90); top.addWidget(self.suffix)
        self.shots_on = QCheckBox('镜头化'); self.shots_on.setChecked(True); top.addWidget(self.shots_on)
        top.addWidget(QLabel('≤'))
        self.max_shot = QDoubleSpinBox(); self.max_shot.setRange(3.0, 30.0); self.max_shot.setValue(8.0)
        self.max_shot.setSuffix(' s'); self.max_shot.setMaximumWidth(90); top.addWidget(self.max_shot)
        top.addWidget(QLabel('切片'))
        self.mode = QComboBox(); self.mode.addItems(['copy', 'encode', 'prores']); top.addWidget(self.mode)
        top.addStretch()
        root.addLayout(top)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['视频', '状态', '结果', '耗时'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setColumnWidth(1, 110); self.table.setColumnWidth(3, 80)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.doubleClicked.connect(self.open_review)
        root.addWidget(self.table, 1)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(140)
        root.addWidget(self.log)
        self.setCentralWidget(central)
        self.timer = self.startTimer(500)                          # pump the queue

    # ---- queue ----
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '选择素材文件夹')
        if not folder:
            return
        videos = scan_videos(folder)
        if not videos:
            self.log.appendPlainText(f'文件夹中没有视频: {folder}')
            return
        for v in videos:
            base = os.path.splitext(os.path.basename(v))[0]
            out = os.path.splitext(v)[0] + self.suffix.text()
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(os.path.basename(v)))
            self.table.setItem(row, 1, QTableWidgetItem('排队中'))
            self.table.setItem(row, 2, QTableWidgetItem(''))
            self.table.setItem(row, 3, QTableWidgetItem(''))
            self.queue.append(dict(video=v, out=out, row=row, t0=None))
        self.log.appendPlainText(f'加入 {len(videos)} 个视频（队列 {len(self.queue)}）')

    def pump(self):
        while len(self.running) < WORKERS and self.queue:
            item = self.queue.pop(0)
            cmd = [sys.executable, '-m', 'dronesift', item['video'],
                   '--mode', self.mode.currentText(), '--out', item['out']]
            if self.shots_on.isChecked():
                cmd += ['--max-shot-s', str(self.max_shot.value())]
            proc = QProcess(self)
            proc.readyReadStandardOutput.connect(lambda p=proc: self.on_output(p))
            proc.finished.connect(lambda code, status, p=proc: self.on_finished(p, code))
            item['t0'] = time.time()
            self.running[proc] = item
            proc.start(cmd[0], cmd[1:])
            self.set_status(item['row'], '处理中…')

    def set_status(self, row, text, col=1):
        self.table.item(row, col).setText(text)

    def on_output(self, proc):
        item = self.running.get(proc)
        if not item:
            return
        text = bytes(proc.readAllStandardOutput()).decode('utf-8', 'replace').strip()
        if text:
            self.log.appendPlainText(f"[{os.path.basename(item['video'])}] {text}")
            for line in text.splitlines():
                if line.startswith('kept frames'):
                    self.set_status(item['row'], line.strip())

    def on_finished(self, proc, code):
        item = self.running.pop(proc, None)
        if not item:
            return
        self.done.append(item)
        dt = time.time() - item['t0']
        self.set_status(item['row'], f'完成 {dt:.0f}s' if code == 0 else f'失败（见日志）', 1)
        if code == 0:
            base = os.path.splitext(os.path.basename(item['video']))[0]
            n = self.count_shots(item['out'], base)
            self.set_status(item['row'], f'{n} 个镜头' if n is not None else '完成', 2)
            self.set_status(item['row'], f'{dt:.0f}s', 3)
        proc.deleteLater()

    @staticmethod
    def count_shots(out, base):
        for name in (base + '_shots.json', base + '_segments.json'):
            p = os.path.join(out, name)
            if os.path.exists(p):
                d = json.load(open(p))
                return len(d.get('shots') or d['segments'])
        return None

    def open_review(self, index):
        name = self.table.item(index.row(), 0).text()
        video = out = None
        for it in self.done + list(self.running.values()):
            if os.path.basename(it['video']) == name:
                video, out = it['video'], it['out']
        if video is None:
            self.log.appendPlainText(f'尚无结果可审查: {name}')
            return
        base = os.path.splitext(os.path.basename(video))[0]
        sj = os.path.join(out, base + '_shots.json'); gj = os.path.join(out, base + '_segments.json')
        src = sj if os.path.exists(sj) else gj
        if not os.path.exists(src):
            self.log.appendPlainText('结果还未生成'); return
        d = json.load(open(src))
        win = ReviewWindow(video, out, d.get('shots') or [], d.get('segments') or [], d['fps'], parent=self)
        win.setAttribute(Qt.WA_DeleteOnClose)
        win.show()
        self._wins = getattr(self, '_wins', []) + [win]

    def timerEvent(self, ev):
        self.pump()

def main():
    if HAVE_WEBENGINE:
        QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
