import shutil
import os
from datetime import datetime
from pathlib import Path

def backup_milvus_db(db_path='milvus_db/milvus.db', backup_dir='backups'):
    """
    Crea un backup de la base de datos Milvus
    """
    # Crear directorio de backups
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(exist_ok=True)
    
    # Verificar que existe la base de datos
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"❌ Base de datos no encontrada: {db_path}")
        return False
    
    # Crear nombre de backup con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"milvus_db_backup_{timestamp}"
    backup_path = backup_dir / backup_name
    
    # Calcular tamaño
    def get_dir_size(path):
        total = 0
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total
    
    db_size = get_dir_size(db_path.parent) if db_path.is_file() else get_dir_size(db_path)
    size_mb = db_size / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print(f"🗄️  BACKUP DE BASE DE DATOS MILVUS")
    print(f"{'='*60}")
    print(f"Origen: {db_path}")
    print(f"Destino: {backup_path}")
    print(f"Tamaño: {size_mb:.2f} MB")
    print(f"{'='*60}\n")
    
    # Confirmar
    response = input("¿Continuar con el backup? (y/n): ")
    if response.lower() != 'y':
        print("❌ Backup cancelado")
        return False
    
    try:
        print("📦 Copiando archivos...")
        
        # Si db_path es un archivo
        if db_path.is_file():
            backup_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(db_path, backup_path / db_path.name)
            # Copiar toda la carpeta que contiene el db
            shutil.copytree(db_path.parent, backup_path, dirs_exist_ok=True)
        else:
            # Si es un directorio
            shutil.copytree(db_path, backup_path)
        
        print(f"✅ Backup completado exitosamente!")
        print(f"📁 Ubicación: {backup_path}")
        
        # Listar backups existentes
        list_backups(backup_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante el backup: {e}")
        return False

def list_backups(backup_dir='backups'):
    """
    Lista todos los backups disponibles
    """
    backup_dir = Path(backup_dir)
    
    if not backup_dir.exists():
        print("\n📂 No hay backups disponibles")
        return
    
    backups = sorted([d for d in backup_dir.iterdir() if d.is_dir()], 
                    key=lambda x: x.stat().st_mtime, 
                    reverse=True)
    
    if not backups:
        print("\n📂 No hay backups disponibles")
        return
    
    print(f"\n{'='*60}")
    print(f"📚 BACKUPS DISPONIBLES ({len(backups)} total)")
    print(f"{'='*60}")
    
    for i, backup in enumerate(backups, 1):
        # Calcular tamaño
        size = sum(f.stat().st_size for f in backup.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        
        # Fecha de creación
        timestamp = datetime.fromtimestamp(backup.stat().st_mtime)
        
        print(f"{i}. {backup.name}")
        print(f"   📅 Fecha: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   💾 Tamaño: {size_mb:.2f} MB")
        print()

def restore_backup(backup_name, db_path='milvus_db/milvus.db', backup_dir='backups'):
    """
    Restaura un backup
    """
    backup_dir = Path(backup_dir)
    backup_path = backup_dir / backup_name
    
    if not backup_path.exists():
        print(f"❌ Backup no encontrado: {backup_path}")
        return False
    
    db_path = Path(db_path)
    
    print(f"\n{'='*60}")
    print(f"⚠️  RESTAURAR BACKUP")
    print(f"{'='*60}")
    print(f"⚠️  ADVERTENCIA: Esto sobrescribirá la base de datos actual!")
    print(f"Origen: {backup_path}")
    print(f"Destino: {db_path}")
    print(f"{'='*60}\n")
    
    response = input("¿Estás SEGURO de querer restaurar? (escriba 'SI' para confirmar): ")
    if response != 'SI':
        print("❌ Restauración cancelada")
        return False
    
    try:
        # Backup de la base actual antes de restaurar
        if db_path.exists():
            print("📦 Haciendo backup de la base de datos actual...")
            backup_current_name = f"milvus_db_before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(db_path.parent, backup_dir / backup_current_name)
            print(f"   ✅ Backup actual guardado en: {backup_current_name}")
        
        # Eliminar base de datos actual
        print("🗑️  Eliminando base de datos actual...")
        if db_path.is_file():
            shutil.rmtree(db_path.parent)
        else:
            shutil.rmtree(db_path)
        
        # Restaurar backup
        print("📥 Restaurando backup...")
        shutil.copytree(backup_path, db_path.parent if db_path.is_file() else db_path)
        
        print(f"\n✅ Backup restaurado exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error durante la restauración: {e}")
        return False

def delete_old_backups(backup_dir='backups', keep_last=5):
    """
    Elimina backups antiguos, manteniendo solo los últimos N
    """
    backup_dir = Path(backup_dir)
    
    if not backup_dir.exists():
        return
    
    backups = sorted([d for d in backup_dir.iterdir() if d.is_dir()], 
                    key=lambda x: x.stat().st_mtime, 
                    reverse=True)
    
    if len(backups) <= keep_last:
        print(f"✅ Solo hay {len(backups)} backups, no es necesario eliminar")
        return
    
    to_delete = backups[keep_last:]
    
    print(f"\n🗑️  Eliminando {len(to_delete)} backups antiguos...")
    for backup in to_delete:
        print(f"   Eliminando: {backup.name}")
        shutil.rmtree(backup)
    
    print(f"✅ Mantenidos los últimos {keep_last} backups")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gestión de backups de Milvus')
    parser.add_argument('action', choices=['backup', 'list', 'restore', 'cleanup'], 
                       help='Acción a realizar')
    parser.add_argument('--db_path', default='milvus_db/milvus.db', 
                       help='Ruta a la base de datos')
    parser.add_argument('--backup_dir', default='backups', 
                       help='Directorio de backups')
    parser.add_argument('--backup_name', help='Nombre del backup para restaurar')
    parser.add_argument('--keep_last', type=int, default=5, 
                       help='Número de backups a mantener')
    
    args = parser.parse_args()
    
    if args.action == 'backup':
        backup_milvus_db(args.db_path, args.backup_dir)
    
    elif args.action == 'list':
        list_backups(args.backup_dir)
    
    elif args.action == 'restore':
        if not args.backup_name:
            print("❌ Error: --backup_name es requerido para restaurar")
            list_backups(args.backup_dir)
        else:
            restore_backup(args.backup_name, args.db_path, args.backup_dir)
    
    elif args.action == 'cleanup':
        delete_old_backups(args.backup_dir, args.keep_last)