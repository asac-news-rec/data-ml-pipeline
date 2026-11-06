# 기여 가이드 (Contributing Guide)

data-ml-pipeline 저장소에 기여해 주셔서 감사합니다.
이 저장소는 ASAC 파이널 프로젝트(뉴스 추천 시스템)의 데이터 처리(Airflow DAG,
BigQuery SQL), 피처, 모델 학습·평가 코드를 관리합니다.
원활한 협업을 위해 아래 규칙을 따라 주세요.

---

## 워크플로우

```
이슈 등록 → 작업 브랜치 생성 → 작업/검증 → Draft PR 생성 → 셀프 리뷰 및 설명 보강
       → Ready for review 전환 → 이해도 체크 inline 답변
       → 팀원 리뷰 요청 → 최소 2명 승인 → Squash merge
```

1. **이슈 등록**: 코드가 변경되는 작업은 반드시 이슈를 먼저 만듭니다.
   `Issues > New issue`에서 Issue Form(Feature / Bug / Experiment)을 선택해 작성해 주세요. Form을 선택하면 제목 prefix와 label이 자동으로 적용됩니다.

2. **작업 브랜치 생성**: 브랜치는 **해당 이슈에서 생성**합니다.
   이슈 우측 `Development > Create a branch`를 사용하면 브랜치가 이슈에 자동 연결되고, `main` 기준으로 분기됩니다. 브랜치 네이밍 규칙은 아래를 따릅니다.
   GitHub이 제안한 기본 이름은 사용하지 말고, 생성 대화상자에서
   `<type>/<이슈번호>-<영어-소문자-슬러그>`로 직접 수정합니다.

3. **작업 및 검증**: 커밋 컨벤션에 따라 커밋 메시지를 작성합니다. 모든 변경은 `git diff --check`를 통과시키고, 변경 영역별 검증(Python [계획]: lint·pytest, DAG [계획]: import 로드 확인)을 실행합니다.

4. **Draft PR 생성**: 작업이 완료되면 Draft PR을 엽니다. PR 템플릿을 채우고 본문에 `Closes #이슈번호`를 포함합니다.

5. **셀프 리뷰 및 설명 보강**: 본인이 diff를 처음부터 끝까지 읽고 PR 설명, 검증 결과, 영향 범위(데이터 계약, BigQuery 비용)를 보강합니다.

6. **Ready for review 전환**: 설명이 충분해지면 Draft를 해제해 리뷰를 요청할 수 있는 상태로 전환합니다.

7. **에이전트 리뷰 실행**: Draft가 아닌 PR을 열거나 Ready for review로 전환하면 `claude.yml`이 에이전트 리뷰를 실행합니다.

8. **이해도 체크 inline 답변**: 에이전트가 남긴 `이해도 확인:` inline comment 각각에 같은 스레드에서 답변하고, 필요하면 로컬에서 검증한 뒤 resolve합니다.

9. **팀원 리뷰 요청**: 담당 팀원에게 리뷰를 요청합니다.

10. **최소 2명 승인**: 팀원 **최소 2명**의 Approve와 모든 대화 resolve가 있어야 머지할 수 있습니다.

11. **Squash merge**: 머지 방식은 **squash만 허용**합니다.
    머지 커밋 제목은 `<type>: <설명> (#PR번호)` 형식으로 작성합니다.

---

## 브랜치 네이밍 규칙

브랜치는 **해당 이슈의 `Create a branch`로 생성**합니다. 생성한 브랜치명을
로컬에서 체크아웃해 작업합니다. Create a branch 대화상자의 기본 제안 이름은
규칙을 만족하지 않을 수 있으므로, 생성 전에 아래 형식의 영어 이름으로 수정합니다.

```bash
# 이슈에서 Create a branch로 브랜치 생성(예: feat/42-add-clean-events-dag) 후
git fetch origin
git switch feat/42-add-clean-events-dag
```

| 유형 | 패턴 | 예시 |
|------|------|------|
| 기능 개발 | `feat/이슈번호-간략한-설명` | `feat/42-add-clean-events-dag` |
| 버그 수정 | `fix/이슈번호-간략한-설명` | `fix/57-fix-feature-window` |
| 실험 | `exp/이슈번호-간략한-설명` | `exp/61-compare-lightgbm-params` |
| 문서 | `docs/이슈번호-간략한-설명` | `docs/30-update-readme` |
| 리팩터링 | `refactor/이슈번호-간략한-설명` | `refactor/48-split-feature-sql` |
| 기타 | `chore/이슈번호-간략한-설명` | `chore/10-setup-ci` |

- 영어 소문자와 하이픈(`-`)만 사용합니다.
- 이슈 번호를 반드시 포함합니다.
- `branch-name-policy` 검사가 형식을 확인합니다. required check로 지정하면 형식 위반 브랜치는 main에 병합할 수 없습니다. (지정 여부는 아래 "main 브랜치 보호 규칙"을 참고합니다.)
- GitHub의 Revert 자동 브랜치(`revert-<PR번호>-...`)는 긴급 롤백을 위해 예외로 허용합니다.

---

## 커밋 컨벤션

```
<type>: <설명>
```

### Type 목록

| type | 사용 상황 |
|------|-----------|
| `feat` | DAG, SQL, 피처, 학습·평가 코드, GitHub 자동화 추가 |
| `fix` | DAG·SQL·코드·workflow·문서 오류 수정 |
| `refactor` | 동작 변화 없는 코드/SQL/문서 구조 정리 |
| `docs` | 문서 추가·수정 |
| `chore` | 저장소 기본 설정, 관리 작업 |
| `exp` | 모델, 피처, 지표 실험 |

### 예시

```
feat: clean_events DAG 추가
fix: 피처 집계 창 경계 오류 수정
exp: 인기순 baseline과 LightGBM 비교
docs: 피처 정의서 초안 작성
```

- 설명은 한국어로 작성합니다.
- 제목은 현재형 동사로 시작합니다 (추가, 수정, 삭제, ...).
- 제목은 50자 이내로 작성합니다.

---

## main 브랜치 보호 규칙

**목표 규칙** (팀이 지키기로 한 것):

- **직접 push 금지**: 모든 변경은 PR을 통해서만 반영됩니다.
- **리뷰 승인 필수**: 최소 2명의 팀원 Approve와 모든 대화 resolve가 있어야 머지할 수 있습니다.
- **PR 통과 후 머지**: 설정한 필수 체크가 모두 통과해야 머지할 수 있습니다.
- **머지 방식**: squash만 허용.

**[확인 필요] 현재 상태**: 2026-11-02 기준 `main`에는 GitHub의 보호 규칙이
설정되어 있지 않습니다. 비공개 레포에서 규칙을 강제할 수 있는지는 GitHub 레포
Settings → Rules에서 확인하고, 강제할 수 없으면 위 규칙을 팀 약속으로 지킵니다.

## 데이터·ML 변경 리뷰 원칙

- 테이블·피처 스키마, 이벤트 필드 같은 데이터 계약이 바뀌면 PR에 적고 영향받는 팀원을 리뷰어로 넣습니다.
- 학습 데이터는 노출 시각 이전 값만 쓰도록 시점 기준으로 조인합니다 (미래 정보 누수 방지).
- BigQuery 쿼리는 파티션 필터를 쓰고 `SELECT *`를 피합니다.
- Secret 값은 코드, 로그, PR 본문에 포함하지 않습니다.
- service account key, `.env`, 생성된 데이터·모델 파일은 커밋하지 않습니다.

---

## Claude 자동 리뷰

자동 리뷰 workflow(`.github/workflows/claude.yml`)는 infra 레포의 파일을 가져와 이 레포에 맞게 고친 것입니다. 시크릿은 `CLAUDE_SETUP_TOKEN`, 모델은 `claude-opus-5-5`를 씁니다.

- 에이전트가 남긴 `이해도 확인:` inline comment에 같은 스레드에서 답변하고 resolve합니다.
- 에이전트 작업 규칙은 `CLAUDE.md`(`AGENTS.md`는 symlink)를 기준으로 합니다.

---

## Label 및 Project 운영

label은 필요할 때 레포에 직접 만들어 사용합니다. 후보: `feature`, `bug`, `experiment`, `documentation`, `chore`, `question`.
Issue Form이 label을 자동으로 붙이려면 레포에 같은 이름의 label이 있어야 합니다.

Project 보드는 아직 사용하지 않습니다. **[미정]**

새 작업은 Issues 화면에서 Issue Form을 선택해 생성합니다.

---

## CI 및 검증

- `.github/workflows/branch-name-policy.yml` — 브랜치 이름 형식을 검사합니다.
- **[계획]** Python 코드가 생기면 lint·pytest·DAG 로드 검사 workflow를 추가합니다.

로컬 검증:

```bash
git diff --check
```

- Python 변경 **[계획]**: 패키지 관리·lint·테스트 도구가 정해지면 명령을 적습니다.
- DAG 변경 **[계획]**: infra 레포 compose의 Airflow에서 DAG가 오류 없이 로드되는지 확인합니다.

---

## 문제 해결

**PR이 merge되지 않을 때**: Draft 상태인지, approve 2명이 있는지, 모든 대화가 resolve되었는지, 충돌이 있는지, required check(`branch-name-policy`)가 실패했는지 확인합니다.

**이슈가 자동으로 닫히지 않을 때**: PR 본문에 `Closes #이슈번호`가 있는지, PR이 `main`으로 merge되었는지, 이슈 번호가 같은 저장소의 번호인지 확인합니다.

---

## 참고 링크

- Repository: https://github.com/asac-news-rec/data-ml-pipeline
- Issue Forms: `.github/ISSUE_TEMPLATE/*.yml`
- PR template: `.github/PULL_REQUEST_TEMPLATE.md`
- 에이전트 가이드: `CLAUDE.md`, `AGENTS.md`(symlink)
