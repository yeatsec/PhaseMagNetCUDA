
#ifndef LINKEDLIST_CUH
#define LINKEDLIST_CUH

#include <stdint.h>
#include <stdio.h>
#include <algorithm>

/* For Debugging purposes */
struct Dyn {
	float* data;
	Dyn() {
		data = nullptr;
	}
	Dyn(const Dyn& toCopy) {
		if (toCopy.data != nullptr) {
			data = new float[1];
			*data = *(toCopy.data);
		}
		else {
			data = nullptr;
		}
	}
	Dyn& operator=(Dyn other) {
		std::swap(data, other.data);
		return *this;
	}

	~Dyn() {
		delete data;
		data = nullptr;
	}
};

template <typename T>
class LinkedListNode {
private:
	T elem;
	LinkedListNode<T>* next;
	LinkedListNode<T>* prev;
public:
	LinkedListNode(T& elem, LinkedListNode<T>* prev = nullptr, LinkedListNode<T>* next = nullptr) :
		elem(elem),
		prev(prev),
		next(next) {};
	LinkedListNode(LinkedListNode<T>& toCopy) :
		elem(toCopy.elem), // copy of elem
		prev(toCopy.prev), // changed from nullptr
		next(toCopy.next) {};
	LinkedListNode<T>& operator=(LinkedListNode<T> rhs) {
		std::swap(elem, rhs.elem);
		std::swap(prev, rhs.prev);
		std::swap(next, rhs.next);
		return *this;
	}
	~LinkedListNode() {
		prev = next = nullptr; // unlink
	};
	T getElem(void) {
		return elem;
	};
	T* getElemPtr(void) {
		return &elem;
	};
	void setNext(LinkedListNode<T>* ptr) {
		next = ptr;
	};
	void setPrev(LinkedListNode<T>* ptr) {
		prev = ptr;
	};
	LinkedListNode<T>* getNext(void) const {
		return next;
	};
	LinkedListNode<T>* getPrev(void) const {
		return prev;
	};
	bool hasNext(void) const {
		return next != nullptr;
	};
	bool hasPrev(void) const {
		return prev != nullptr;
	};
};

template <typename T>
class LinkedList {
private:
	LinkedListNode<T>* head;
	LinkedListNode<T>* tail;
	size_t size;
public:
	LinkedList() :
		head(nullptr),
		tail(nullptr),
		size(0) {};
	LinkedList(const LinkedList<T>& toCopy) {
		size = 0;
		head = nullptr;
		tail = nullptr;
		// walk toCopy's linkedList and append
		if (!toCopy.isEmpty()) {
			for (const LinkedListNode<T>* walkptr = toCopy.head; walkptr->hasNext(); walkptr = walkptr->next) {
				append(walkptr->getElem());
			}
		}
	}
	LinkedList<T>& operator=(LinkedList<T> rhs) {
		// copy-and-swap
		std::swap(size, rhs.size);
		std::swap(head, rhs.head);
		std::swap(tail, rhs.tail);
		return *this;
	}
	~LinkedList() {
		if (!isEmpty()) {
			for (LinkedListNode<T>* walkptr = head; walkptr->hasNext(); walkptr = walkptr->getNext()) {
				delete (walkptr->getPrev()); // nop if nullptr
			}
			delete tail;
			tail = nullptr;
			head = nullptr;
		}
	};
	void append(T elem) {
		// edge condition: head / tail are nullptr, size 0
		if (size == 0) {
			head = new LinkedListNode<T>(elem);
			tail = head;
		}
		else {
			tail->setNext(new LinkedListNode<T>(elem, tail));
			tail = tail->getNext();
		}
		++size;
	};
	LinkedListNode<T>* getHead(void) const {
		return head;
	}; // heh
	LinkedListNode<T>* getTail(void) const {
		return tail;
	}; // heh
	size_t getSize(void) const {
		return size;
	}
	bool isEmpty(void) const {
		return size == 0;
	};
	void forEach(void func(LinkedListNode<T>*)) {
		for (LinkedListNode<T>* ptr = getHead(); ptr->hasNext(); ptr = ptr->getNext()) // will end at tail
			func(ptr);
		func(getTail()); // don't forget the tail
	}
	void forEachReverse(void func(LinkedListNode<T>*)) {
		for (LinkedListNode<T>* ptr = getTail(); ptr->hasPrev(); ptr = ptr->getPrev())
			func(ptr);
		func(getHead()); // don't forget the head
	}
};


#endif // LINKEDLIST_CUH