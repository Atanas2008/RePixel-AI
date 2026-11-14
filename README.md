# RePixel-AI — етап 1: снимки

Програмата използва готови модели Real-ESRGAN; не обучава нов модел. В този
етап работят снимки през командния ред. Видео, възстановяване на лица и
графичен интерфейс все още не са реализирани.

## 1. Инсталирайте Python

Инсталирайте **Python 3.11** от [python.org](https://www.python.org/downloads/).
В Windows отметнете „Add Python to PATH“. Отворете Terminal (macOS/Linux) или
PowerShell (Windows) в папката на проекта. Проверете версията:

```text
python --version
```

На macOS/Linux може да е нужно да използвате `python3.11` вместо `python`.

## 2. Създайте виртуална среда

Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Ако PowerShell забрани активирането, изпълнете
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` и повторете
командата за активиране.

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Когато средата е активна, командата `python` по-долу сочи към нея. Python
3.10 не се поддържа от избраната поправена версия на Pillow.

## 3. Инсталирайте библиотеките

Първо подгответе инструментите и NumPy:

```bash
python -m pip install --upgrade pip
python -m pip install setuptools==80.9.0 wheel==0.46.3 Cython==3.3.1 numpy==1.26.4
```

След това инсталирайте PyTorch според компютъра:

* **NVIDIA, Windows/Linux:** инсталирайте актуален NVIDIA драйвер и изпълнете:

  ```bash
  python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
  ```

  При проблем с драйвера проверете вариантите на
  [pytorch.org](https://pytorch.org/get-started/locally/).
* **Без NVIDIA, Windows/Linux:** за работа с процесора изпълнете:

  ```bash
  python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
  ```

* **macOS:** изпълнете:

  ```bash
  python -m pip install torch==2.10.0 torchvision==0.25.0
  ```

  На Apple Silicon програмата използва MPS, когато е наличен.

Накрая инсталирайте останалите библиотеки:

```bash
python -m pip install --no-build-isolation -r requirements.txt
```

Първоначалната инсталация може да отнеме време и място на диска. BasicSR
изисква предварително инсталираните по-горе инструменти; опцията
`--no-build-isolation` ги използва вместо да изтегля второ копие на PyTorch.

## 4. Пробвайте снимка

Поставете например `photo.jpg` в папката на проекта и изпълнете:

```bash
python enhance.py photo.jpg --scale 2
```

При първо стартиране моделът се изтегля от официалния проект в `models/`;
необходим е интернет. Резултатът е `output/photo_enhanced.jpg`. Оригиналът
остава непроменен; ако резултатът вече съществува, се добавя номер.
Папките `models` и `output` се създават автоматично.

Още примери:

```bash
python enhance.py photo.png --scale 4 --model realesr-general-x4v3
python enhance.py photos --scale 2 --model realesr-animevideov3 --tile 128
```

Вторият пример обработва всички JPG, PNG и WEBP файлове непосредствено в
папката `photos`. Моделите са `RealESRGAN_x4plus` (подробни снимки),
`realesr-general-x4v3` (по-бърз) и `realesr-animevideov3` (аниме/рисунки).
Резултатите могат да се различават според снимката; възстановяването не
гарантира оригиналните детайли. Максималният размер е 3840 × 2160 пиксела,
като пропорциите се запазват; големите снимки се намаляват при нужда.
Програмата показва избраното устройство: CUDA, MPS или CPU.

## 5. Проблеми

* **AMD/Intel видеокарта:** тази версия използва CPU, ако CUDA или MPS не са
  налични. За Vulkan ускорение разгледайте отделния проект
  [realesrgan-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN#portable-executable-files-ncnn);
  той не е интегриран тук.
* **Недостатъчна памет:** опитайте `--tile 128` или `--tile 64`. Обработката
  е на блокове, но самите входно и изходно изображение също заемат RAM.
* **`functional_tensor`:** BasicSR използва премахнат модул на torchvision.
  `core/models.py` локално го насочва към съвместимата функция на torchvision;
  не е нужно да редактирате инсталираните библиотеки.
* **Грешка при изтегляне:** проверете интернет връзката и повторете командата.
  Незавършеното изтегляне се изтрива автоматично.
* **`No module named ...`:** активирайте `.venv` и повторете стъпка 3.

Тестовете се пускат с `python -m unittest discover -s tests -v`. Те използват
заместващ модел и не изтеглят AI тегла.
