# data-ml-pipeline
Data pipeline (ingest, clean, features) and ML training

## 에이전트 하네스

AI 코딩 도구가 이 저장소에서 일할 때 따르는 규칙은 `CLAUDE.md`(`AGENTS.md`는 링크)에
있고, 사람용 협업 규칙은 `CONTRIBUTING.md`에 있습니다.
이 규칙들은 우리 [infra](https://github.com/asac-news-rec/infra) 레포의 하네스를 가져와
이 레포에 맞게 줄이고 수정한 것입니다 (경량 구성: 상세 가이드 폴더 `.claude/docs/`는 아직 넣지 않음).

### 클론 위치: WSL 또는 Linux·macOS

`AGENTS.md`는 `CLAUDE.md`를 가리키는 symlink입니다. Windows Git으로 받으면(`core.symlinks=false`)
링크 대신 `CLAUDE.md`라는 한 줄만 든 일반 파일이 만들어져, `AGENTS.md`를 읽는 도구(OpenCode 등)가
규칙을 받지 못합니다. 그래서 이 레포는 **WSL의 리눅스 폴더(`~/`)에 클론**합니다.
`/mnt/c/...` 같은 Windows 폴더는 쓰지 않습니다.

확인 방법:

```bash
ls -l AGENTS.md    # WSL·Linux·macOS: AGENTS.md -> CLAUDE.md 가 보이면 정상
```

```powershell
Get-Content AGENTS.md    # Windows: CLAUDE.md 한 줄만 나오면 링크가 풀린 상태
```

링크가 풀렸다면 WSL에서 다시 클론합니다. Windows에서 그대로 작업해야 하면 `CLAUDE.md`를 직접
읽게 하고, 풀린 `AGENTS.md`는 커밋하지 않습니다.
