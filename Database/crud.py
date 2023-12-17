from sqlalchemy.orm import Session
from . import models, schemas


def get_index_path(db: Session, index_id: str):
    return db.query(models.Index).filter(models.Index.indexId == index_id).first()


def create_index(db: Session, index: schemas.IndexCreate):
    db_index = models.Index(indexId=index.indexId, indexPath=index.indexPath)
    db.add(db_index)
    db.commit()
    db.refresh(db_index)
    return db_index
