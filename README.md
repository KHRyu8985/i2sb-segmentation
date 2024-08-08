### 프로젝트 구조
```
i2sb-segmentation/
├── Pipfile
├── Pipfile.lock
├── README.md
├── data/
│   ├── OCTA500_3M
│   ├── OCTA500_3M.dvc
│   ├── OCTA500_6M
│   ├── OCTA500_6M.dvc
│   ├── ROSSA
│   └── ROSSA.dvc
├── nbs/
│   └── test.ipynb
├── script/
│   ├── exp.py
│   └── unit_test/
└── src/
    ├── __init__.py
    ├── archs/
    ├── data/
    ├── losses/
    ├── metrics/
    ├── models/
    └── utils/
```

### 설치 및 실행 방법

1. Pipenv 설치 (아직 설치하지 않은 경우):
   ```
   pip install pipenv
   ```

2. 프로젝트 클론 및 디렉토리 이동:
   ```
   git clone [repository_url]
   cd i2sb-segmentation
   ```

3. Pipenv 환경 생성 및 의존성 설치:
   ```
   pipenv install
   ```

4. Pipenv 환경 활성화:
   ```
   pipenv shell
   ```

5. 실험 스크립트 실행 (예시: 데이터로더와 metric이 잘 작동하는지 코드):
   ```
   python script/unit_test/test_segmentation_2d_dataset.py
   ```

6. Jupyter Notebook 실행 (필요한 경우):
   ```
   pipenv run jupyter notebook
   ```

### 데이터

- OCTA500_3M, OCTA500_6M, ROSSA 데이터셋을 사용.
- 데이터는 DVC(Data Version Control)로 관리.

### 주요 디렉토리 설명

- `data/`: 데이터셋 저장 위치
- `nbs/`: Jupyter Notebook 파일 저장 위치
- `script/`: 실험 및 테스트 스크립트 저장 위치
- `src/`: 소스 코드 저장 위치
  - `archs/`: 모델 아키텍처 관련 코드
  - `data/`: 데이터 처리 관련 코드
  - `losses/`: 손실 함수 관련 코드
  - `metrics/`: 평가 지표 관련 코드
  - `models/`: 모델 구현 관련 코드
  - `utils/`: 유틸리티 함수 저장 위치

### DVC 사용법 (데이터 백업)

1. DVC 초기화 (본 리포지토리에서 init 되어있으므로 생략):
   ```
   dvc init
   ```

2. 원격 저장소 추가:
   ```
   dvc remote add -d myremote ssh://user@host:/path/to/dvc-backup
   ```

3. 원격 저장소 설정 수정:
   ```
   dvc remote modify myremote user username
   dvc remote modify myremote password password
   ```

4. 데이터 추적 (본 리포지토리에서 add 되어있으므로 생략):
   ```
   dvc add data/dataset_name
   ```

5. 변경사항 커밋 (본 리포지토리에서 commit 되어있으므로 생략):
   ```
   git add .dvc
   git commit -m "Add data to DVC"
   ```

6. 데이터 푸시(본 리포지토리에서는 pull 목적이므로 생략):
   ```
   dvc push
   ```

7. 데이터 풀 (이 부분만 실행):
   ```
   dvc pull
   ```

8. 데이터 상태 확인:
   ```
   dvc status
   ```

9. 특정 버전의 데이터 가져오기:
   ```
   git checkout <commit-hash>
   dvc checkout
   ```