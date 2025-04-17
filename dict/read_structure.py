
from rich.console import Console
from rich.tree import Tree

"""
이 모듈은 JSON 데이터를 Python dict 형식으로 받아,
rich 라이브러리의 Tree 객체를 이용해 해당 데이터 구조를 시각적으로 표현하고자 할 때
유용한 함수를 제공합니다.

주요 함수:
- build_tree(trials, tree): 주어진 trials(dict, list 등)를 재귀적으로 순회하며,
  Rich의 Tree 객체에 노드를 추가합니다.
- create_tree(trials, root_label): 주어진 dict 데이터를 바탕으로 루트 노드를 생성한 뒤,
  build_tree 함수를 이용해 전체 구조를 구성한 Tree 객체를 반환합니다.
- print_tree(trials, root_label): create_tree 함수를 통해 생성한 Tree 객체를
  콘솔에 출력합니다.

사용 예시:
1) JSON 파일을 읽어 dict 형태로 로드합니다.
2) print_tree 함수를 호출해 구조를 확인합니다.
   (필요에 따라 create_tree를 직접 호출한 뒤, console.print를 사용할 수도 있습니다.)

	output_dir = "output"
	json_file = f"{output_dir}/2pv7_confidences.json"
	with open(json_file, "r") as f:
		trials = json.load(f)
	print_tree(trials)


의존성:
- json
- rich
  - from rich.tree import Tree
  - from rich.console import Console

모듈을 직접 실행할 경우, "output" 디렉터리 내의 2pv7_confidences.json 파일을 로드하여
해당 JSON 구조를 Tree 형태로 출력합니다.
"""


def build_tree(data, tree: Tree) -> None:
	"""
	주어진 data의 구조를 rich의 Tree 객체에 재귀적으로 추가합니다.

	인자:
		trials: 탐색할 데이터 (dict, list, 또는 기타 타입)
		tree (Tree): 현재 노드
	"""
	if isinstance(data, dict):
		for key, value in data.items():
			if isinstance(value, dict):
				branch = tree.add(f"[bold]{key}[/bold]/")
				build_tree(value, branch)
			elif isinstance(value, list):
				branch = tree.add(f"[bold]{key}[/bold]: <list> (길이: {len(value)})")
				# 리스트의 첫 번째 요소가 dict인 경우, 바로 구조를 추가합니다.
				if value and isinstance(value[0], dict):
					build_tree(value[0], branch)
			else:
				tree.add(f"[bold]{key}[/bold]: {type(value).__name__}")
	elif isinstance(data, list):
		branch = tree.add(f"<list> (길이: {len(data)})")
		if data and isinstance(data[0], dict):
			build_tree(data[0], branch)
	else:
		tree.add(str(data))

def create_tree(data: dict, root_label: str = "[bold green]JSON 구조[/bold green]") -> Tree:
	"""
	주어진 dict 데이터를 기반으로 rich의 Tree 객체를 생성하여 반환합니다.

	인자:
		trials (dict): JSON 데이터로 변환된 dict
		root_label (str): 트리의 루트 노드에 표시할 레이블 (기본값: "[bold green]JSON 구조[/bold green]")

	반환:
		Tree: 데이터 구조를 나타내는 rich Tree 객체
	"""
	tree = Tree(root_label)
	build_tree(data, tree)
	return tree


def print_tree(data: dict, root_label: str = "[bold green]JSON 구조[/bold green]") -> None:
	"""
	주어진 dict 데이터를 기반으로 rich의 Tree 객체를 생성하고 출력합니다.

	인자:
		trials (dict): JSON 데이터로 변환된 dict
		root_label (str): 트리의 루트 노드에 표시할 레이블 (기본값: "[bold green]JSON 구조[/bold green]")
	"""
	tree = create_tree(data, root_label)
	console = Console()
	console.print(tree)



if __name__ == "__main__":
	output_dir = "output"
	json_file = f"{output_dir}/2pv7_confidences.json"
	with open(json_file, "r") as f:
		data = json.load(f)
	print_tree(data)