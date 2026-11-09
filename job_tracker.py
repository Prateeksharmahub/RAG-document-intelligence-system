"""
Job Tracker Module
In-memory job tracking with CSV persistence
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading


class JobTracker:
    """
    In-memory job tracker with CSV persistence
    Thread-safe for concurrent access
    """
    
    def __init__(self, csv_file: str = "jobs.csv"):
        """
        Initialize job tracker
        
        Args:
            csv_file: Path to CSV file for persistence
        """
        self.csv_file = Path(csv_file)
        self.jobs: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        
        # CSV columns
        self.columns = [
            'job_id',
            'collection',
            'status',
            'message',
            'created_at',
            'completed_at',
            'total_files',
            'processed_files',
            'total_chunks',
            'error'
        ]
        
        # Load existing jobs from CSV if file exists
        self.load_from_csv()
    
    def create_job(
        self,
        job_id: str,
        collection: str,
        status: str = "queued",
        message: str = "Job queued"
    ) -> Dict:
        """
        Create a new job
        
        Args:
            job_id: Unique job identifier
            collection: Collection name
            status: Initial status
            message: Initial message
        
        Returns:
            Job dictionary
        """
        with self.lock:
            job = {
                'job_id': job_id,
                'collection': collection,
                'status': status,
                'message': message,
                'created_at': datetime.now().isoformat(),
                'completed_at': None,
                'total_files': 0,
                'processed_files': 0,
                'total_chunks': 0,
                'error': None
            }
            
            self.jobs[job_id] = job
            self.save_to_csv()
            
            return job.copy()
    
    def update_job(
        self,
        job_id: str,
        **kwargs
    ) -> Optional[Dict]:
        """
        Update job fields
        
        Args:
            job_id: Job identifier
            **kwargs: Fields to update
        
        Returns:
            Updated job dictionary or None if not found
        """
        with self.lock:
            if job_id not in self.jobs:
                return None
            
            # Update fields
            for key, value in kwargs.items():
                if key in self.jobs[job_id]:
                    self.jobs[job_id][key] = value
            
            self.save_to_csv()
            
            return self.jobs[job_id].copy()
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """
        Get job by ID
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job dictionary or None if not found
        """
        with self.lock:
            return self.jobs.get(job_id, None)
    
    def get_all_jobs(self) -> List[Dict]:
        """
        Get all jobs
        
        Returns:
            List of all job dictionaries
        """
        with self.lock:
            return list(self.jobs.values())
    
    def get_jobs_by_status(self, status: str) -> List[Dict]:
        """
        Get jobs by status
        
        Args:
            status: Job status to filter by
        
        Returns:
            List of job dictionaries with matching status
        """
        with self.lock:
            return [
                job for job in self.jobs.values()
                if job['status'] == status
            ]
    
    def get_jobs_by_collection(self, collection: str) -> List[Dict]:
        """
        Get jobs by collection name
        
        Args:
            collection: Collection name to filter by
        
        Returns:
            List of job dictionaries with matching collection
        """
        with self.lock:
            return [
                job for job in self.jobs.values()
                if job['collection'] == collection
            ]
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job
        
        Args:
            job_id: Job identifier
        
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                self.save_to_csv()
                return True
            return False
    
    def get_stats(self) -> Dict:
        """
        Get job statistics
        
        Returns:
            Dictionary with job statistics
        """
        with self.lock:
            total = len(self.jobs)
            
            stats = {
                'total_jobs': total,
                'queued': 0,
                'processing': 0,
                'completed': 0,
                'failed': 0
            }
            
            for job in self.jobs.values():
                status = job['status']
                if status in stats:
                    stats[status] += 1
            
            return stats
    
    def save_to_csv(self):
        """Save jobs to CSV file"""
        try:
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
                
                for job in self.jobs.values():
                    # Write only the columns we care about
                    row = {col: job.get(col, '') for col in self.columns}
                    writer.writerow(row)
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    def load_from_csv(self):
        """Load jobs from CSV file"""
        if not self.csv_file.exists():
            return
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    job_id = row['job_id']
                    
                    # Convert empty strings to None
                    job = {}
                    for key, value in row.items():
                        if value == '':
                            job[key] = None
                        elif key in ['total_files', 'processed_files', 'total_chunks']:
                            job[key] = int(value) if value else 0
                        else:
                            job[key] = value
                    
                    self.jobs[job_id] = job
            
            print(f"Loaded {len(self.jobs)} jobs from {self.csv_file}")
        except Exception as e:
            print(f"Error loading from CSV: {e}")
    
    def export_to_csv(self, output_file: str):
        """
        Export jobs to a specific CSV file
        
        Args:
            output_file: Path to output CSV file
        """
        with self.lock:
            try:
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.columns)
                    writer.writeheader()
                    
                    for job in self.jobs.values():
                        row = {col: job.get(col, '') for col in self.columns}
                        writer.writerow(row)
                
                return True
            except Exception as e:
                print(f"Error exporting to CSV: {e}")
                return False
    
    def clear_completed_jobs(self):
        """Clear all completed jobs"""
        with self.lock:
            to_delete = [
                job_id for job_id, job in self.jobs.items()
                if job['status'] == 'completed'
            ]
            
            for job_id in to_delete:
                del self.jobs[job_id]
            
            self.save_to_csv()
            
            return len(to_delete)
    
    def clear_all_jobs(self):
        """Clear all jobs"""
        with self.lock:
            count = len(self.jobs)
            self.jobs.clear()
            self.save_to_csv()
            return count


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = JobTracker("test_jobs.csv")
    
    # Create a job
    job = tracker.create_job(
        job_id="job-001",
        collection="test_collection",
        status="queued",
        message="Job created"
    )
    print(f"Created job: {job}")
    
    # Update job
    tracker.update_job(
        "job-001",
        status="processing",
        message="Processing files...",
        total_files=5,
        processed_files=2
    )
    print(f"\nUpdated job: {tracker.get_job('job-001')}")
    
    # Create another job
    tracker.create_job(
        job_id="job-002",
        collection="another_collection",
        status="completed",
        message="Job completed"
    )
    
    # Get stats
    stats = tracker.get_stats()
    print(f"\nJob statistics: {stats}")
    
    # Get all jobs
    all_jobs = tracker.get_all_jobs()
    print(f"\nAll jobs ({len(all_jobs)}):")
    for job in all_jobs:
        print(f"  - {job['job_id']}: {job['status']}")
    
    # Export to CSV
    tracker.export_to_csv("exported_jobs.csv")
    print("\nExported jobs to exported_jobs.csv")

