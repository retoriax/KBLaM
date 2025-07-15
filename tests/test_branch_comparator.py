"""
Test cases for the branch comparator tool.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sys
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from kblam.utils.branch_comparator import BranchComparator, FileDifference, Task


class TestBranchComparator(unittest.TestCase):
    """Test cases for BranchComparator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.comparator = BranchComparator()
    
    def test_file_difference_creation(self):
        """Test FileDifference dataclass creation."""
        diff = FileDifference(
            file_path="test.py",
            change_type="added",
            diff_content="test content",
            new_size=100
        )
        
        self.assertEqual(diff.file_path, "test.py")
        self.assertEqual(diff.change_type, "added")
        self.assertEqual(diff.diff_content, "test content")
        self.assertEqual(diff.new_size, 100)
        self.assertIsNone(diff.old_size)
    
    def test_task_creation(self):
        """Test Task dataclass creation."""
        task = Task(
            task_id="TEST-001",
            title="Test task",
            description="Test description",
            file_path="test.py",
            change_type="modified"
        )
        
        self.assertEqual(task.task_id, "TEST-001")
        self.assertEqual(task.title, "Test task")
        self.assertEqual(task.priority, "medium")  # default value
    
    def test_generate_tasks_for_added_file(self):
        """Test task generation for added files."""
        differences = [
            FileDifference(
                file_path="new_file.py",
                change_type="added",
                diff_content="def new_function(): pass",
                new_size=25
            )
        ]
        
        tasks = self.comparator.generate_tasks(differences, "branch1", "branch2")
        
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.change_type, "added")
        self.assertEqual(task.priority, "high")
        self.assertIn("new_file.py", task.title)
        self.assertIn("branch2", task.description)
    
    def test_generate_tasks_for_deleted_file(self):
        """Test task generation for deleted files."""
        differences = [
            FileDifference(
                file_path="deleted_file.py",
                change_type="deleted",
                diff_content="def old_function(): pass",
                old_size=25
            )
        ]
        
        tasks = self.comparator.generate_tasks(differences, "branch1", "branch2")
        
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.change_type, "deleted")
        self.assertEqual(task.priority, "high")
        self.assertIn("deleted_file.py", task.title)
        self.assertIn("branch1", task.description)
    
    def test_generate_tasks_for_modified_file(self):
        """Test task generation for modified files."""
        differences = [
            FileDifference(
                file_path="modified_file.py",
                change_type="modified",
                diff_content="- old line\n+ new line",
                old_size=20,
                new_size=30
            )
        ]
        
        tasks = self.comparator.generate_tasks(differences, "branch1", "branch2")
        
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.change_type, "modified")
        self.assertEqual(task.priority, "medium")
        self.assertIn("modified_file.py", task.title)
        self.assertIn("20 -> 30", task.description)
    
    def test_save_tasks_to_file(self):
        """Test saving tasks to JSON file."""
        tasks = [
            Task(
                task_id="TEST-001",
                title="Test task",
                description="Test description",
                file_path="test.py",
                change_type="added",
                priority="high"
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.comparator.save_tasks_to_file(tasks, temp_file)
            
            # Read back the file and verify
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]['task_id'], "TEST-001")
            self.assertEqual(data[0]['title'], "Test task")
            self.assertEqual(data[0]['change_type'], "added")
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    @patch('subprocess.run')
    def test_run_git_command_success(self, mock_run):
        """Test successful git command execution."""
        mock_result = Mock()
        mock_result.stdout = "test output"
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        result = self.comparator._run_git_command(["git", "status"])
        
        self.assertEqual(result, "test output")
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_run_git_command_failure(self, mock_run):
        """Test git command failure handling."""
        from subprocess import CalledProcessError
        
        mock_run.side_effect = CalledProcessError(1, "git", stderr="error")
        
        with self.assertRaises(RuntimeError):
            self.comparator._run_git_command(["git", "invalid"])


if __name__ == '__main__':
    unittest.main()