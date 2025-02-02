
from rich.tree import Tree

import json
from rich.console import Console


def build_tree(data, tree: Tree) -> None:
	"""
	주어진 data의 구조를 rich의 Tree 객체에 재귀적으로 추가합니다.

	인자:
		data: 탐색할 데이터 (dict, list, 또는 기타 타입)
		tree (Tree): 현재 노드
	"""
	if isinstance(data, dict):
		for key, value in data.items():
			if isinstance(value, dict):
				branch = tree.add(f"[bold]{key}[/bold]/")
				build_tree(value, branch)
			elif isinstance(value, list):
				branch = tree.add(f"[bold]{key}[/bold]: <list> (길이: {len(value)})")
				# 리스트의 첫 번째 요소가 dict인 경우 예시 구조를 추가합니다.
				if value and isinstance(value[0], dict):
					example_branch = branch.add("(예시 요소 구조:)")
					build_tree(value[0], example_branch)
			else:
				tree.add(f"[bold]{key}[/bold]: {type(value).__name__}")
	elif isinstance(data, list):
		branch = tree.add(f"<list> (길이: {len(data)})")
		# 리스트의 첫 번째 요소가 dict인 경우 예시 구조를 추가합니다.
		if data and isinstance(data[0], dict):
			example_branch = branch.add("(예시 요소 구조:)")
			build_tree(data[0], example_branch)
	else:
		tree.add(str(data))

def create_tree(data: dict, root_label: str = "[bold green]JSON 구조[/bold green]") -> Tree:
	"""
	주어진 dict 데이터를 기반으로 rich의 Tree 객체를 생성하여 반환합니다.

	인자:
		data (dict): JSON 데이터로 변환된 dict
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
		data (dict): JSON 데이터로 변환된 dict
		root_label (str): 트리의 루트 노드에 표시할 레이블 (기본값: "[bold green]JSON 구조[/bold green]")
	"""
	tree = create_tree(data, root_label)
	console = Console()
	console.print(tree)


