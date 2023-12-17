from pydantic import BaseModel


class IndexBase(BaseModel):
    indexId: str
    indexPath: str


class IndexCreate(IndexBase):
    pass


class Index(IndexBase):
    id: int

    class Config:
        orm_mode = True
